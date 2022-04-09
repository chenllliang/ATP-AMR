from pathlib import Path

import torch
import pdb
try:
    from torch.cuda.amp import autocast
    autocast_available = True
except ImportError:
    class autocast:
        def __init__(self, enabled=True): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_value, exc_traceback): pass
    autocast_available = False

from torch.cuda.amp.grad_scaler import GradScaler
import transformers

from spring_amr import ROOT
from spring_amr.dataset import reverse_direction
from spring_amr.optim import RAdam,ChildRAdam
from spring_amr.evaluation import write_predictions, compute_smatch, predict_amrs, predict_sentences, compute_bleu,compute_bleu_from_files
from spring_amr.utils import instantiate_model_and_tokenizer, instantiate_model_and_tokenizer_multitask,instantiate_loader,instantiate_multitask_loader,instantiate_real_multitask_loader

from tqdm import tqdm

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, global_step_from_engine

from ranger21 import Ranger21


def do_multitask_pretrain(checkpoint=None, direction='amr', split_both_decoder=False, fp16=False):
    assert direction in ["multi_amr","multi"]
    train_mode = config['train_mode']
    model, tokenizer = instantiate_model_and_tokenizer_multitask(
        config['model'],
        checkpoint=checkpoint,
        additional_tokens_smart_init=config['smart_init'],
        dropout=config['dropout'],
        attention_dropout=config['attention_dropout'],
        from_pretrained=config['warm_start'],
        init_reverse=split_both_decoder,
        penman_linearization=config['penman_linearization'],
        collapse_name_ops=config['collapse_name_ops'],
        use_pointer_tokens=config['use_pointer_tokens'],
        raw_graph=config.get('raw_graph', False)
    )

    print(model)
    print(model.config)

    # HACK Add Child Tuning Optimizer
    if train_mode in ["ChildTuning-D","ChildTuning-F"]:
        optimizer = ChildRAdam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            reserve_p = config['reserve_p'],
            mode = train_mode
        )
        print("using %s"%train_mode)
    else:
        optimizer = RAdam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'])

    # if checkpoint is not None:
    #     optimizer.load_state_dict(torch.load(checkpoint)['optimizer'])

    if config['scheduler'] == 'cosine':
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=config['training_steps'])
    elif config['scheduler'] == 'constant':
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'])
    else:
        raise ValueError

    scaler = GradScaler(enabled=fp16)

    train_src_paths = {"amr":config['amr_train'],"ndp":config['dp_train_src'],"ssrl":config["srl_train_src"]}
    train_golds_paths = {"amr":config['amr_train'],"ndp":config["dp_train_src"],"ssrl":config["srl_train_gold"]}

    dev_src_paths = {"amr":config['amr_dev']}
    dev_src_paths = {"amr":config['amr_dev']}

    train_loader = instantiate_real_multitask_loader(
        direction,
        train_src_paths,
        train_golds_paths,
        tokenizer,
        batch_size=config['batch_size'],
        remove_longer_than = config['remove_longer_than'],
        evaluation=False,
    )

    print("There are {} batches in a epoch".format(len(train_loader)))

    dev_gold_path = ROOT / 'data/tmp/dev-gold_multi.txt'
    dev_pred_path = ROOT / 'data/tmp/dev-pred_multi.txt'

    dev_loader = instantiate_real_multitask_loader(
        direction,
        dev_src_paths,
        dev_src_paths,
        tokenizer,out=dev_gold_path,
        batch_size=config['batch_size'],
        remove_longer_than = config['remove_longer_than'],
        evaluation=True,
    )

    # caculate gradient mask and set gradient mask
    if train_mode == "ChildTuning-D":

        gradient_mask = dict()
        model.cuda()
        device = next(model.parameters()).device
        train_loader.device = device

        model.train()

        for name, params in model.named_parameters():
            if 'layers' in name:
                gradient_mask[params] = params.new_zeros(params.size())

        N = len(train_loader)
        print(N)

        for batch in tqdm(train_loader):
            x, y, extra = batch
            model.amr_mode = True
            loss, *_ = model(**x, **y)
            loss.backward()

            for name, params in model.named_parameters():
                if 'layers' in name:
                    torch.nn.utils.clip_grad_norm_(params, 1)
                    gradient_mask[params] += (params.grad ** 2) / N
            
            model.zero_grad()

        print('Calculate Fisher Information')

        # Numpy
        r = None
        for k, v in gradient_mask.items():
            v = v.view(-1).cpu().numpy()
            if r is None:
                r = v
            else:
                r = np.append(r, v)
        
        polar = np.percentile(r, (1-config['reserve_p'])*100)
        for k in gradient_mask:
            gradient_mask[k] = gradient_mask[k] >= polar
        print('Polar => {}'.format(polar))

        # TODO: pytorch: torch.kthvalue
        
        optimizer.set_gradient_mask(gradient_mask)
    


    if direction == 'multi':
        def train_step(engine, batch):
            model.train()
            x, y, extra = batch
            model.amr_mode = True
            with autocast(enabled=fp16):
                loss, *_ = model(**x, **y)
                scaler.scale((loss / config['accum_steps'])).backward()
                return loss.item()

        @torch.no_grad()
        def eval_step(engine, batch):
            model.eval()
            x, y, extra = batch
            model.amr_mode = True
            loss, *_ = model(**x, **y)
            return loss.item()

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    @trainer.on(Events.STARTED)
    def update(engine):
        print('training started!')
    

    @trainer.on(Events.EPOCH_COMPLETED)
    @trainer.on(Events.ITERATION_COMPLETED(every=config['accum_steps']))
    def update(engine):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if config['train_mode'] != "ranger":
            # ranger is coupled with scheduler
            scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_trn_loss(engine):
        log_msg = f"training epoch: {engine.state.epoch}"
        if direction in ('multi', 'multi_amr'):
            log_msg += f" | loss_amr: {engine.state.metrics['trn_amr_loss']:.3f}"
        print(log_msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_dev_eval(engine):
        dev_loader.batch_size = config['batch_size']
        dev_loader.device = next(model.parameters()).device
        evaluator.run(dev_loader)

    if not config['best_loss']:
        if direction in ('multi', 'multi_amr'):
            @evaluator.on(Events.EPOCH_COMPLETED)
            def smatch_eval(engine):
                device = next(model.parameters()).device
                dev_loader.device = device
                graphs = predict_amrs(dev_loader, model, tokenizer, restore_name_ops=config['collapse_name_ops'])
                write_predictions(dev_pred_path, tokenizer, graphs)
                try:
                    smatch = compute_smatch(dev_gold_path, dev_pred_path)
                except:
                    smatch = 0.
                engine.state.metrics['dev_smatch'] = smatch


    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_dev_loss(engine):
        log_msg = f"dev epoch: {trainer.state.epoch}"
        if direction in ('multi', 'multi_amr'):
            log_msg += f" | loss_amr: {engine.state.metrics['dev_amr_loss']:.3f}"
            if not config['best_loss']:
                log_msg += f" | smatch: {engine.state.metrics['dev_smatch']:.3f}"

        print(log_msg)

    if direction in ['multi','multi_amr']:
        RunningAverage(output_transform=lambda out: out).attach(trainer, 'trn_amr_loss')
        RunningAverage(output_transform=lambda out: out).attach(evaluator, 'dev_amr_loss')


    if config['save_checkpoints']:

        if direction in ('multi', 'multi_amr'):
            if config['best_loss']:
                prefix = 'best-loss-amr'
                score_function = lambda x: 1 / evaluator.state.metrics['dev_amr_loss']
            else:
                prefix = 'best-smatch'
                score_function = lambda x: evaluator.state.metrics['dev_smatch']
        else:
            if config['best_loss']:
                prefix = 'best-loss-text'
                score_function = lambda x: 1 / evaluator.state.metrics['dev_amr_loss']
            else:
                prefix = 'best-bleu'
                score_function = lambda x: evaluator.state.metrics['dev_bleu']

        to_save = {'model': model, 'optimizer': optimizer}
        if config['log_wandb']:
            where_checkpoints = str(wandb_logger.run.dir)
        else:
            root = ROOT/'runs'
            try:
                root.mkdir()
            except:
                pass
            where_checkpoints = root/str(str(len(list(root.iterdir())))+"_"+str(config['name']))
            try:
                where_checkpoints.mkdir()
            except:
                pass
            where_checkpoints = str(where_checkpoints)

        print(where_checkpoints)
        handler = ModelCheckpoint(
            where_checkpoints,
            prefix,
            n_saved=5,
            create_dir=True,
            score_function=score_function,
            global_step_transform=global_step_from_engine(trainer),
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)

    model.cuda()
    device = next(model.parameters()).device
    train_loader.device = device
    trainer.run(train_loader, max_epochs=config['max_epochs'])








def do_single_pretrain(checkpoint=None, direction='amr', split_both_decoder=False, fp16=False):
    assert direction in ('dp','srl','ner','sa','mt')
    train_mode = config['train_mode']
    

    model, tokenizer = instantiate_model_and_tokenizer(
        config['model'],
        checkpoint=checkpoint,
        additional_tokens_smart_init=config['smart_init'],
        dropout=config['dropout'],
        attention_dropout=config['attention_dropout'],
        from_pretrained=config['warm_start'],
        init_reverse=split_both_decoder,
        penman_linearization=config['penman_linearization'],
        collapse_name_ops=config['collapse_name_ops'],
        use_pointer_tokens=config['use_pointer_tokens'],
        raw_graph=config.get('raw_graph', False)
    )

    
    print(model)
    print(model.config)

    # HACK Add Child Tuning Optimizer
    if train_mode in ["ChildTuning-D","ChildTuning-F"]:
        optimizer = ChildRAdam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            reserve_p = config['reserve_p'],
            mode = train_mode
        )
        print("using %s"%train_mode)
    else:
        optimizer = RAdam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'])

    # if checkpoint is not None:
    #     optimizer.load_state_dict(torch.load(checkpoint)['optimizer'])

    if config['scheduler'] == 'cosine':
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=config['training_steps'])
    elif config['scheduler'] == 'constant':
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'])
    else:
        raise ValueError

    scaler = GradScaler(enabled=fp16)

    train_loader = instantiate_multitask_loader(
        direction,
        config['train_src'],
        config['train_gold'],
        tokenizer,
        batch_size=config['batch_size'],
        remove_longer_than = config['remove_longer_than'],
        evaluation=False,
    )

    dev_loader = instantiate_multitask_loader(
        direction,
        config['dev_src'],
        config['dev_gold'],
        tokenizer,
        batch_size=config['batch_size'],
        remove_longer_than = config['remove_longer_than'],
        evaluation=True,
    )

    #pdb.set_trace()


    # caculate gradient mask and set gradient mask
    if train_mode == "ChildTuning-D":

        gradient_mask = dict()
        model.cuda()
        device = next(model.parameters()).device
        train_loader.device = device

        model.train()

        for name, params in model.named_parameters():
            if 'layers' in name:
                gradient_mask[params] = params.new_zeros(params.size())

        N = len(train_loader)
        print(N)

        for batch in tqdm(train_loader):
            x, y, extra = batch
            model.amr_mode = True
            loss, *_ = model(**x, **y)
            loss.backward()

            for name, params in model.named_parameters():
                if 'layers' in name:
                    torch.nn.utils.clip_grad_norm_(params, 1)
                    gradient_mask[params] += (params.grad ** 2) / N
            
            model.zero_grad()

        print('Calculate Fisher Information')

        # Numpy
        r = None
        for k, v in gradient_mask.items():
            v = v.view(-1).cpu().numpy()
            if r is None:
                r = v
            else:
                r = np.append(r, v)
        
        polar = np.percentile(r, (1-config['reserve_p'])*100)
        for k in gradient_mask:
            gradient_mask[k] = gradient_mask[k] >= polar
        print('Polar => {}'.format(polar))

        # TODO: pytorch: torch.kthvalue
        
        optimizer.set_gradient_mask(gradient_mask)


    if direction == 'dp':

        def train_step(engine, batch):
            model.train()
            x, y, extra = batch
            model.amr_mode = True
            with autocast(enabled=fp16):
                loss, *_ = model(**x, **y)
            scaler.scale((loss / config['accum_steps'])).backward()
            return loss.item()

        @torch.no_grad()
        def eval_step(engine, batch):
            model.eval()
            x, y, extra = batch
            model.amr_mode = True
            loss, *_ = model(**x, **y)
            return loss.item()
        
    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    @trainer.on(Events.STARTED)
    def update(engine):
        print('training started!')

    @trainer.on(Events.EPOCH_COMPLETED)

    @trainer.on(Events.ITERATION_COMPLETED(every=config['accum_steps']))
    def update(engine):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_trn_loss(engine):
        log_msg = f"training epoch: {engine.state.epoch}"
        if direction in ('amr', 'both'):
            log_msg += f" | loss_amr: {engine.state.metrics['trn_amr_loss']:.3f}"
        if direction in ('text', 'both'):
            log_msg += f" | loss_text: {engine.state.metrics['trn_text_loss']:.3f}"
        if direction in ('dp'):
            log_msg += f" | loss_dp: {engine.state.metrics['trn_dp_loss']:.3f}"
        print(log_msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_dev_eval(engine):
        dev_loader.batch_size = config['batch_size']
        dev_loader.device = next(model.parameters()).device
        evaluator.run(dev_loader)

    if not config['best_loss']:
        if direction in ("dp"):
            @evaluator.on(Events.EPOCH_COMPLETED)
            def smatch_eval(engine):
                device = next(model.parameters()).device
                dev_loader.device = device
                pred_sentences = predict_sentences(dev_loader, model, tokenizer, outdir=str(where_checkpoints+"/"+str(config['name'])+"out_dev_"+str(trainer.state.epoch)), beam_size=config['beam_size'])

                bleu = compute_bleu_from_files(config['dev_gold'],str(where_checkpoints+"/"+str(config['name'])+"out_dev_"+str(trainer.state.epoch)))
                
                engine.state.metrics['dev_bleu'] = bleu.score
    
    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_dev_loss(engine):
        log_msg = f"dev epoch: {trainer.state.epoch}"
        if direction in ('dp'):
            log_msg += f" | loss_dp: {engine.state.metrics['dev_dp_loss']:.3f}"
            if not config['best_loss']:
                log_msg += f" | bleu: {engine.state.metrics['dev_bleu']:.3f}"
        print(log_msg)

    if direction == 'dp':
        RunningAverage(output_transform=lambda out: out).attach(trainer, 'trn_dp_loss')
        RunningAverage(output_transform=lambda out: out).attach(evaluator, 'dev_dp_loss')

    if config['save_checkpoints']:

        if direction in ('dp'):
            if config['best_loss']:
                prefix = 'best-loss-dp'
                score_function = lambda x: 1 / evaluator.state.metrics['dev_dp_loss']
            else:
                prefix = 'best-bleu'
                score_function = lambda x: evaluator.state.metrics['dev_bleu']
       

        to_save = {'model': model, 'optimizer': optimizer}
        if config['log_wandb']:
            where_checkpoints = str(wandb_logger.run.dir)
        else:
            root = ROOT/'runs'
            try:
                root.mkdir()
            except:
                pass
            where_checkpoints = root/str(str(len(list(root.iterdir())))+"_"+str(config['name']))
            try:
                where_checkpoints.mkdir()
            except:
                pass
            where_checkpoints = str(where_checkpoints)

        print(where_checkpoints)
        handler = ModelCheckpoint(
            where_checkpoints,
            prefix,
            n_saved=5,
            create_dir=True,
            score_function=score_function,
            global_step_transform=global_step_from_engine(trainer),
            require_empty=False
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)

    model.cuda()
    device = next(model.parameters()).device
    train_loader.device = device
    trainer.run(train_loader, max_epochs=config['max_epochs'])






def do_train(checkpoint=None, direction='amr', split_both_decoder=False, fp16=False):

    assert direction in ('amr', 'text' ,'both')

    train_mode = config['train_mode']

    model, tokenizer = instantiate_model_and_tokenizer(
        config['model'],
        checkpoint=checkpoint,
        additional_tokens_smart_init=config['smart_init'],
        dropout=config['dropout'],
        attention_dropout=config['attention_dropout'],
        from_pretrained=config['warm_start'],
        init_reverse=split_both_decoder,
        penman_linearization=config['penman_linearization'],
        collapse_name_ops=config['collapse_name_ops'],
        use_pointer_tokens=config['use_pointer_tokens'],
        raw_graph=config.get('raw_graph', False)
    )

    train_loader = instantiate_loader(
        config['train'],
        tokenizer,
        batch_size=config['batch_size'],
        evaluation=False,
        use_recategorization=config['use_recategorization'],
        remove_longer_than=config['remove_longer_than'],
        remove_wiki=config['remove_wiki'],
        dereify=config['dereify'],
    )

    dev_gold_path = ROOT / 'data/tmp/dev-gold.txt'
    dev_pred_path = ROOT / 'data/tmp/dev-pred.txt'

    num_batches = len(train_loader)
    total_iterations = num_batches*config['max_epochs']
    print("There are {} batches in a epoch".format(num_batches))
    print("Compute gradients every {} batches".format(config['accum_steps']))
    print("Total Update Interations = {}".format(total_iterations//config['accum_steps']))


    dev_loader = instantiate_loader(
        config['dev'],
        tokenizer,
        batch_size=config['batch_size'],
        evaluation=True, out=dev_gold_path,
        use_recategorization=config['use_recategorization'],
        remove_wiki=config['remove_wiki'],
        dereify=config['dereify'],
    )

    print(model.config)

    if checkpoint is not None:
        print(f'Checkpoint restored ({checkpoint})!')

    
    if direction == 'both' and split_both_decoder:
        params_dir_enc = list(model.model.encoder.parameters())
        params_dir_enc_check = {id(p) for p in params_dir_enc}
        params_dir_dec = set()
        params_dir_dec |= {p for p in model.model.decoder.parameters() if id(p) not in params_dir_enc_check}
        params_dir_dec |= {p for p in model.rev.model.decoder.parameters() if id(p) not in params_dir_enc_check}
        params_dir_dec = list(params_dir_dec)
        optimizer = RAdam(
            [{'params': params_dir_enc, 'lr': config['learning_rate']},
             {'params': params_dir_dec, 'lr': config['learning_rate'] * 2},],
            weight_decay=config['weight_decay'])
    
    # HACK Add Child Tuning Optimizer
    elif train_mode in ["ChildTuning-D","ChildTuning-F"]:
        optimizer = ChildRAdam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            reserve_p = config['reserve_p'],
            mode = train_mode
        )
        print("using %s"%train_mode)
    
    elif train_mode == "ranger":
        optimizer = Ranger21( 
            model.parameters(),
            lr=0.0005,
            num_epochs=config['max_epochs'],
            num_batches_per_epoch=num_batches//config['accum_steps'],
            use_warmup=True,
            num_warmup_iterations=1000
           )
        print("Init Learning Rate:{}".format(optimizer.current_lr))
    
    # HACK end

    

    else:
        optimizer = RAdam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'])
    


    # if checkpoint is not None:
    #     optimizer.load_state_dict(torch.load(checkpoint)['optimizer'])

    

   

   

    if config['scheduler'] == 'cosine':
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=total_iterations//config['accum_steps'])
    elif config['scheduler'] == 'constant':
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'])
    elif config['scheduler'] == 'linear':
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=total_iterations//config['accum_steps'])
    else:
        raise ValueError

    scaler = GradScaler(enabled=fp16)



    # caculate gradient mask and set gradient mask
    if train_mode == "ChildTuning-D":

        gradient_mask = dict()
        model.cuda()
        device = next(model.parameters()).device
        train_loader.device = device

        model.train()

        for name, params in model.named_parameters():
            if 'layers' in name:
                gradient_mask[params] = params.new_zeros(params.size())

        N = len(train_loader)
        print(N)

        for batch in tqdm(train_loader):
            x, y, extra = batch
            model.amr_mode = True
            loss, *_ = model(**x, **y)
            loss.backward()

            for name, params in model.named_parameters():
                if 'layers' in name:
                    torch.nn.utils.clip_grad_norm_(params, 1)
                    gradient_mask[params] += (params.grad ** 2) / N
            
            model.zero_grad()

        print('Calculate Fisher Information')

        # Numpy
        r = None
        for k, v in gradient_mask.items():
            v = v.view(-1).cpu().numpy()
            if r is None:
                r = v
            else:
                r = np.append(r, v)
        
        polar = np.percentile(r, (1-config['reserve_p'])*100)
        for k in gradient_mask:
            gradient_mask[k] = gradient_mask[k] >= polar
        print('Polar => {}'.format(polar))

        # TODO: pytorch: torch.kthvalue
        
        optimizer.set_gradient_mask(gradient_mask)
    


    if direction == 'amr':
        def train_step(engine, batch):
            model.train()
            x, y, extra = batch
            model.amr_mode = True
            with autocast(enabled=fp16):
                loss, *_ = model(**x, **y)
                scaler.scale((loss / config['accum_steps'])).backward()
                return loss.item()

        @torch.no_grad()
        def eval_step(engine, batch):
            model.eval()
            x, y, extra = batch
            model.amr_mode = True
            loss, *_ = model(**x, **y)
            return loss.item()

    elif direction == 'text':

        def train_step(engine, batch):
            model.train()
            x, y, extra = batch
            x, y = reverse_direction(x, y)
            model.rev.amr_mode = False
            with autocast(enabled=fp16):
                loss, *_ = model.rev(**x, **y)
            scaler.scale((loss / config['accum_steps'])).backward()
            return loss.item()

        @torch.no_grad()
        def eval_step(engine, batch):
            model.eval()
            x, y, extra = batch
            x, y = reverse_direction(x, y)
            model.rev.amr_mode = False
            loss, *_ = model(**x, **y)
            return loss.item()

    elif direction == 'both':

        def train_step(engine, batch):
            model.train()
            x, y, extra = batch
            model.amr_mode = True
            with autocast(enabled=fp16):
                loss1, *_ = model(**x, **y)
            scaler.scale((loss1 / config['accum_steps'] * 0.5)).backward()
            loss1 = loss1.item()
            x, y = reverse_direction(x, y)
            model.rev.amr_mode = False
            with autocast(enabled=fp16):
                loss2, *_ = model.rev(**x, **y)
            scaler.scale((loss2 / config['accum_steps'] * 0.5)).backward()
            return loss1, loss2.item()

        @torch.no_grad()
        def eval_step(engine, batch):
            model.eval()
            x, y, extra = batch
            model.amr_mode = True
            loss1, *_ = model(**x, **y)
            x, y = reverse_direction(x, y)
            model.rev.amr_mode = False
            loss2, *_ = model.rev(**x, **y)
            return loss1.item(), loss2.item()

    else:
        raise ValueError

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    @trainer.on(Events.STARTED)
    def update(engine):
        print('training started!')


    

    @trainer.on(Events.EPOCH_COMPLETED)
    @trainer.on(Events.ITERATION_COMPLETED(every=config['accum_steps']))
    def update(engine):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if train_mode!=["ranger"]:
            scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_trn_loss(engine):
        log_msg = f"training epoch: {engine.state.epoch}"
        if direction in ('amr', 'both'):
            log_msg += f" | loss_amr: {engine.state.metrics['trn_amr_loss']:.3f}"
        if direction in ('text', 'both'):
            log_msg += f" | loss_text: {engine.state.metrics['trn_text_loss']:.3f}"
        print(log_msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_dev_eval(engine):
        dev_loader.batch_size = config['batch_size']
        dev_loader.device = next(model.parameters()).device
        evaluator.run(dev_loader)

    if not config['best_loss']:
        if direction in ('amr', 'both'):
            @evaluator.on(Events.EPOCH_COMPLETED)
            def smatch_eval(engine):
                device = next(model.parameters()).device
                dev_loader.device = device
                graphs = predict_amrs(dev_loader, model, tokenizer, restore_name_ops=config['collapse_name_ops'])
                write_predictions(dev_pred_path, tokenizer, graphs)
                try:
                    smatch = compute_smatch(dev_gold_path, dev_pred_path)
                except:
                    smatch = 0.
                engine.state.metrics['dev_smatch'] = smatch

        if direction in ('text', 'both'):
            @evaluator.on(Events.EPOCH_COMPLETED)
            def smatch_eval(engine):
                device = next(model.parameters()).device
                dev_loader.device = device
                pred_sentences = predict_sentences(dev_loader, model.rev, tokenizer, beam_size=config['beam_size'])
                bleu = compute_bleu(dev_loader.dataset.sentences, pred_sentences)
                engine.state.metrics['dev_bleu'] = bleu.score

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_dev_loss(engine):
        log_msg = f"dev epoch: {trainer.state.epoch}"
        if direction in ('amr', 'both'):
            log_msg += f" | loss_amr: {engine.state.metrics['dev_amr_loss']:.3f}"
            if not config['best_loss']:
                log_msg += f" | smatch: {engine.state.metrics['dev_smatch']:.3f}"
        if direction in ('text', 'both'):
            log_msg += f" | loss_text: {engine.state.metrics['dev_text_loss']:.3f}"
            if not config['best_loss']:
                log_msg += f" | bleu: {engine.state.metrics['dev_bleu']:.3f}"
        print(log_msg)

    if direction == 'amr':
        RunningAverage(output_transform=lambda out: out).attach(trainer, 'trn_amr_loss')
        RunningAverage(output_transform=lambda out: out).attach(evaluator, 'dev_amr_loss')
    elif direction == 'text':
        RunningAverage(output_transform=lambda out: out).attach(trainer, 'trn_text_loss')
        RunningAverage(output_transform=lambda out: out).attach(evaluator, 'dev_text_loss')
    elif direction == 'both':
        RunningAverage(output_transform=lambda out: out[0]).attach(trainer, 'trn_amr_loss')
        RunningAverage(output_transform=lambda out: out[1]).attach(trainer, 'trn_text_loss')
        RunningAverage(output_transform=lambda out: out[0]).attach(evaluator, 'dev_amr_loss')
        RunningAverage(output_transform=lambda out: out[1]).attach(evaluator, 'dev_text_loss')


    if config['log_wandb']:
        from ignite.contrib.handlers.wandb_logger import WandBLogger
        wandb_logger = WandBLogger(init=False)

        if direction == 'amr':
            wandb_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="iterations/trn_amr_loss",
                output_transform=lambda loss: loss
            )
        elif direction == 'text':
            wandb_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="iterations/trn_text_loss",
                output_transform=lambda loss: loss
            )
        if direction == 'both':
            wandb_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="iterations/trn_amr_loss",
                output_transform=lambda loss: loss[0]
            )
            wandb_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="iterations/trn_text_loss",
                output_transform=lambda loss: loss[1]
            )

        if direction == 'amr':
            metric_names_trn = ['trn_amr_loss']
            metric_names_dev = ['dev_amr_loss']
            if not config['best_loss']:
                metric_names_dev.append('dev_smatch')
        elif direction == 'text':
            metric_names_trn = ['trn_text_loss']
            metric_names_dev = ['dev_text_loss']
            if not config['best_loss']:
                metric_names_dev.append('dev_bleu')
        elif direction == 'both':
            metric_names_trn = ['trn_amr_loss', 'trn_text_loss']
            metric_names_dev = ['dev_amr_loss', 'dev_smatch']
            if not config['best_loss']:
                metric_names_dev.extend(['dev_text_loss', 'dev_bleu'])

        wandb_logger.attach_output_handler(
            trainer,
            event_name=Events.EPOCH_COMPLETED,
            tag="epochs",
            metric_names=metric_names_trn,
            global_step_transform=lambda *_: trainer.state.iteration,
        )

        wandb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="epochs",
            metric_names=metric_names_dev,
            global_step_transform=lambda *_: trainer.state.iteration,
        )

        @trainer.on(Events.ITERATION_COMPLETED)
        def wandb_log_lr(engine):
            wandb.log({'lr': scheduler.get_last_lr()[0]}, step=engine.state.iteration)

    if config['save_checkpoints']:

        if direction in ('amr', 'both'):
            if config['best_loss']:
                prefix = 'best-loss-amr'
                score_function = lambda x: 1 / evaluator.state.metrics['dev_amr_loss']
            else:
                prefix = 'best-smatch'
                score_function = lambda x: evaluator.state.metrics['dev_smatch']
        else:
            if config['best_loss']:
                prefix = 'best-loss-text'
                score_function = lambda x: 1 / evaluator.state.metrics['dev_amr_loss']
            else:
                prefix = 'best-bleu'
                score_function = lambda x: evaluator.state.metrics['dev_bleu']

        to_save = {'model': model, 'optimizer': optimizer}
        if config['log_wandb']:
            where_checkpoints = str(wandb_logger.run.dir)
        else:
            root = ROOT/'runs'
            try:
                root.mkdir()
            except:
                pass
            where_checkpoints = root/str(str(len(list(root.iterdir())))+"_"+str(config['name']))
            try:
                where_checkpoints.mkdir()
            except:
                pass
            where_checkpoints = str(where_checkpoints)

        print(where_checkpoints)
        handler = ModelCheckpoint(
            where_checkpoints,
            prefix,
            n_saved=5,
            create_dir=True,
            score_function=score_function,
            global_step_transform=global_step_from_engine(trainer),
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)

    model.cuda()
    device = next(model.parameters()).device
    train_loader.device = device
    trainer.run(train_loader, max_epochs=config['max_epochs'])

import random,os
import numpy as np
def seed_torch(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import yaml

    import wandb

    parser = ArgumentParser(
        description="Trainer script",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--direction', type=str, default='amr', choices=['dp','amr', 'text', 'both',"multi","multi_amr"],
        help='Train a uni- (amr, text) or bidirectional (both).')
    parser.add_argument('--split-both-decoder', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/sweeped.yaml',
        help='Use the following config for hparams.')
    parser.add_argument('--checkpoint', type=str,
        help='Warm-start from a previous fine-tuned checkpoint.')
    parser.add_argument('--fp16', action='store_true')
    args, unknown = parser.parse_known_args()

    if args.fp16 and autocast_available:
        raise ValueError('You\'ll need a newer PyTorch version to enable fp16 training.')

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    if config['log_wandb']:
        wandb.init(
            entity="SOME-RUNS",
            project="SOME-PROJECT",
            config=config,
            dir=str(ROOT / 'runs/'))
        config = wandb.config

    print(config)

    seed_torch(int(config['seed']))

    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = None


    


    if args.direction in ["dp"]:
        do_single_pretrain(
                    checkpoint=checkpoint,
        direction=args.direction,
        split_both_decoder=args.split_both_decoder,
        fp16=args.fp16,
        )
    elif args.direction in["multi","amr_multi"]:
        do_multitask_pretrain(
            checkpoint=checkpoint,
            direction=args.direction,
            split_both_decoder=args.split_both_decoder,
            fp16=args.fp16,
        )
    else:
        do_train(
            checkpoint=checkpoint,
            direction=args.direction,
            split_both_decoder=args.split_both_decoder,
            fp16=args.fp16,
        )