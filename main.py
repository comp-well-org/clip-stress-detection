import os
import copy
import json
import time
import numpy as np
import torch
import pandas as pd
import argparse
import warnings
from sklearn import metrics
from constant import EXPS_PATH
from utils import AvgMeter
from utils import get_lifesnaps_loader, get_pmdata_loader
from models import CLIP, BertEncoder, LinearProbe, SupBaselineNet
from models import TransformerEncoder, ResNetSeqEncoder, LSTMEncoder, CNNSeqEncoder
from adapted.cw.simclr import SimCLR
from adapted.cw.byol import BYOL
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    FeatureAblation,
)

warnings.filterwarnings('ignore')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def decompose_batch(batch, dataset_name):
    if dataset_name == 'lifesnaps':
        fitbit, tabular, stress, desc = batch
        fitbit = fitbit.float()
        tabular = tabular.unsqueeze(1).float()
        desc = desc.float()
    elif dataset_name == 'pmdata':
        fitbit, tabular, stress, desc = batch
        fitbit = fitbit.float()
        tabular = tabular.unsqueeze(1).float()
        desc = desc.float()
    return fitbit, tabular, stress, desc

def get_desc_by_dataset(dataset_name):
    descs = [
        'label: relaxed',
        'label: stressed',
    ]
    return descs

def train_loop(
    train_loader, eval_loader, model,
    n_epochs, lr,
    dataset_name, mode='clip',
    save_weights=False,
    save_path=None,
):
    assert mode in ['clip', 'sup', 'leaves', 'simclr', 'byol']
    assert dataset_name in ['lifesnaps', 'pmdata']    
    if mode == 'sup':
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_fn = loss_fn.to(torch.device(DEVICE))
    
    best_model = None
    
    model.to(torch.device(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    results = []
    best_auc = 0
    val_acc = 0
    val_auc = 0
    for epoch in range(n_epochs):
        model.train()
        train_loss = AvgMeter()
        for i, batch in enumerate(train_loader):
            common, tabular, stress, desc = decompose_batch(batch, dataset_name)
            common = common.float().to(torch.device(DEVICE))
            tabular = tabular.float().to(torch.device(DEVICE))
            stress = stress.long().to(torch.device(DEVICE))
            desc = desc.to(torch.device(DEVICE))
            optimizer.zero_grad()
            if mode == 'leaves':  # need dual optimizer if we use leaves
                optimizer_aug = torch.optim.Adam(model.parameters(), lr=lr)
            if mode == 'clip':
                loss = model(desc, common, tabular)
            elif mode in ['simclr', 'byol', 'leaves']:
                loss = model(common)
            elif mode == 'sup':
                # only for indices with valid stress
                nan_indices = stress == -1
                stress = stress[~nan_indices]
                common = common[~nan_indices]
                tabular = tabular[~nan_indices]
                if len(common) == 0:
                    continue
                y_pred = model(common, tabular)
                loss = loss_fn(y_pred, stress)
            
            if mode == 'leaves':
                encoder_loss = loss
                view_maker_loss = -loss.clone()
                optimizer.zero_grad()
                optimizer_aug.zero_grad()
                
                for param in model.seq_encoder.parameters():
                    param.requires_grad = False
                view_maker_loss.backward(retain_graph=True)
                for param in model.seq_encoder.parameters():
                    param.requires_grad = True
                
                for param in model.view.parameters():
                    param.requires_grad = False
                encoder_loss.backward()
                for param in model.view.parameters():
                    param.requires_grad = True
                
                optimizer_aug.step()
                optimizer.step()
                train_loss.update(encoder_loss.item())
            else:
                loss.backward()
                optimizer.step()
                train_loss.update(loss.item())
            msg = f'epoch: {epoch + 1}, batch: {i + 1}/{len(train_loader)}, '
            msg += f'loss: {train_loss.avg:.4f} val_acc: {val_acc:.4f}, val_auc: {val_auc:.4f}'
            if epoch == n_epochs - 1 and i == len(train_loader) - 1:
                print(msg)
            else:
                print(msg, end='\r')
        
        # evaluate
        model.eval()
        val_labels = np.array([])
        val_preds = np.array([])
        with torch.no_grad():
            for _, batch in enumerate(eval_loader):
                # have data
                common, tabular, stress, _ = decompose_batch(batch, dataset_name)
                # move to device
                common = common.float().to(torch.device(DEVICE))
                tabular = tabular.float().to(torch.device(DEVICE))
                stress = stress.long().to(torch.device(DEVICE))
                nan_indices = stress == -1
                stress = stress[~nan_indices]
                common = common[~nan_indices]
                tabular = tabular[~nan_indices]
                if len(common) == 0:
                    continue
                # prediction 
                if mode == 'clip':
                    eval_descs = get_desc_by_dataset(dataset_name)
                    seq_features = model.seq_encoder(common, tabular)
                    seq_embeddings = model.seq_proj(seq_features)
                    text_features = model.text_encoder(eval_descs)
                    text_embeddings = model.text_proj(text_features)
                    logits = (text_embeddings @ seq_embeddings.T) / model.temperature
                    y_pred = logits.argmax(dim=0).cpu().numpy()
                    y_true = stress.cpu().numpy()
                    val_labels = np.concatenate([val_labels, y_true])
                    val_preds = np.concatenate([val_preds, y_pred])
                elif mode == 'sup':
                    y_pred = model(common, tabular).argmax(dim=1).cpu().numpy()
                    y_true = stress.cpu().numpy()
                    val_labels = np.concatenate([val_labels, y_true])
                    val_preds = np.concatenate([val_preds, y_pred])
                elif mode in ['simclr', 'byol', 'leaves']:
                    y_true = stress.cpu().numpy()
                    y_pred = 1 - y_true
                    val_labels = np.concatenate([val_labels, y_true])
                    val_preds = np.concatenate([val_preds, y_pred])
        
        # update the validation accuracy and AUC
        val_acc = (val_preds == val_labels).mean()
        val_auc = metrics.roc_auc_score(val_labels, val_preds)
        
        # save to results
        results.append({
            'epoch': epoch,
            'train_loss': train_loss.avg,
            'val_acc': val_acc,
            'val_auc': val_auc,
        })
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(save_path, 'results.csv'), index=False)
        
        # save the best model
        if mode in ['clip', 'sup']:
            if val_auc > best_auc:
                best_auc = val_auc
                best_model = copy.deepcopy(model)
                if save_weights:
                    torch.save(
                        model.state_dict(), os.path.join(save_path, 'model.pth'),
                    )
        elif mode in ['simclr', 'byol', 'leaves']:
            if epoch == n_epochs - 1:
                best_model = copy.deepcopy(model)
                if save_weights:
                    torch.save(
                        model.state_dict(), os.path.join(save_path, 'model.pth'),
                    )
    
    return best_model

def finetune_clip_loop(
    train_loader, eval_loader, model,
    n_epochs, lr,
    dataset_name,
    save_weights=False, freeze=False,
    save_path=None,
):
    assert dataset_name in ['lifesnaps', 'pmdata']    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(torch.device(DEVICE))
    
    # load the pretrained model and have the probe
    if save_weights:
        model.load_state_dict(torch.load(os.path.join(save_path, 'model.pth')))
    model = LinearProbe(model, out_dim=2, freeze=freeze)
    best_model = None
    
    # finetune the probe
    model.to(torch.device(DEVICE))
    optimizer = torch.optim.Adam(
        [
            {'params': model.seq_encoder.parameters(), 'lr': lr},
            {'params': model.seq_proj.parameters(), 'lr': lr},
            {'params': model.linear.parameters(), 'lr': lr},
        ],
        lr=lr,
    )
    
    # train loop
    results = []
    best_auc = 0
    val_acc = 0
    val_auc = 0
    for epoch in range(n_epochs):
        model.train()
        train_loss = AvgMeter()
        for i, batch in enumerate(train_loader):
            common, tabular, stress, _ = decompose_batch(batch, dataset_name)
            common = common.float().to(torch.device(DEVICE))
            tabular = tabular.float().to(torch.device(DEVICE))
            stress = stress.long().to(torch.device(DEVICE))
            optimizer.zero_grad()
            # only for indices with valid stress
            nan_indices = stress == -1
            stress = stress[~nan_indices]
            common = common[~nan_indices]
            tabular = tabular[~nan_indices]
            if len(common) == 0:
                continue
            y_pred = model(common, tabular)
            loss = loss_fn(y_pred, stress)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item())
            msg = f'epoch: {epoch + 1}, batch: {i + 1}/{len(train_loader)}, '
            msg += f'loss: {train_loss.avg:.4f} val_acc: {val_acc:.4f}, val_auc: {val_auc:.4f}'
            if epoch == n_epochs - 1 and i == len(train_loader) - 1:
                print(msg)
            else:
                print(msg, end='\r')

        # evaluate
        model.eval()
        fa = FeatureAblation(model)
        val_labels = np.array([])
        val_preds = np.array([])
        common_fa = []
        tabular_fa = []
        with torch.no_grad():
            for _, batch in enumerate(eval_loader):
                # have data
                common, tabular, stress, _ = decompose_batch(batch, dataset_name)
                # move to device
                common = common.float().to(torch.device(DEVICE))
                tabular = tabular.float().to(torch.device(DEVICE))
                stress = stress.long().to(torch.device(DEVICE))
                nan_indices = stress == -1
                stress = stress[~nan_indices]
                common = common[~nan_indices]
                tabular = tabular[~nan_indices]
                if len(common) == 0:
                    continue
                # prediction 
                y_pred = model(common, tabular).argmax(dim=1).cpu().numpy()
                y_true = stress.cpu().numpy()
                val_labels = np.concatenate([val_labels, y_true])
                val_preds = np.concatenate([val_preds, y_pred])
                
                if epoch == n_epochs - 1 and not freeze:
                    # feature ablation
                    attributions = fa.attribute(
                        (common, tabular), target=1,
                    )
                    common_fa.append(attributions[0].cpu().numpy())
                    tabular_fa.append(attributions[1].cpu().numpy())

        if epoch == n_epochs - 1 and not freeze:
            common_fa = np.concatenate(common_fa)
            tabular_fa = np.concatenate(tabular_fa)
            common_fa = common_fa.mean(axis=0)
            tabular_fa = tabular_fa.mean(axis=0)
        
            print()
            print('common feature importance:', common_fa.shape)
            print('tabular feature importance:', tabular_fa.shape)
        
            # save as npy
            np.save(os.path.join(save_path, 'common_fa.npy'), common_fa)
            np.save(os.path.join(save_path, 'tabular_fa.npy'), tabular_fa)
        
        # update the validation accuracy and AUC
        val_acc = (val_preds == val_labels).mean()
        val_auc = metrics.roc_auc_score(val_labels, val_preds)
        
        # save to results
        results.append({
            'epoch': epoch,
            'train_loss': train_loss.avg,
            'val_acc': val_acc,
            'val_auc': val_auc,
        })
        results_df = pd.DataFrame(results)
        if freeze:
            results_name = 'results_lp.csv'
            model_path_name = 'model_lp.pth'
        else:
            results_name = 'results_ft.csv'
            model_path_name = 'model_ft.pth'
        results_df.to_csv(os.path.join(save_path, results_name), index=False)
        
        # save the best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = copy.deepcopy(model)
            if save_weights:
                torch.save(
                    model.state_dict(), 
                    os.path.join(save_path, model_path_name),
                )
    return best_model

def get_model(
    seq_enc,
    tab_enc,
    dataset_name,
    mode,
    hidden_size,
    n_layers,
):
    if torch.cuda.is_available():
        device_name = 'cuda'
    else:
        device_name = 'cpu'
    text_encoder = BertEncoder(device=device_name)
    
    if seq_enc == 'transformer':
        seq_net = TransformerEncoder(
            input_size=24, hidden_size=hidden_size, num_layers=n_layers, nhead=4, dropout=0.1,
        )
    elif seq_enc == 'lstm':
        seq_net = LSTMEncoder(
            input_size=24, hidden_size=hidden_size, num_layers=n_layers, dropout=0.1,
        )
    elif seq_enc == 'cnn':
        if dataset_name == 'lifesnaps':
            in_channels = 5
        elif dataset_name == 'pmdata':
            in_channels = 4
        seq_net = CNNSeqEncoder(
            in_channels=in_channels, hidden_size_half=hidden_size // 2, num_layers=n_layers,
        )
    elif seq_enc == 'resnet':
        if dataset_name == 'lifesnaps':
            in_channels = 5
        elif dataset_name == 'pmdata':
            in_channels = 4
        seq_net = ResNetSeqEncoder(
            in_channels=in_channels, base_filters=hidden_size * 2, n_block=n_layers, hidden_size=24,
        )
    if dataset_name == 'lifesnaps':
        input_size = 21
        n_head = 3
    elif dataset_name == 'pmdata':
        input_size = 38
        n_head = 2
    if tab_enc == 'transformer':
        tab_net = TransformerEncoder(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=n_layers, 
            nhead=n_head, dropout=0.1,
        )
    elif tab_enc == 'lstm':
        tab_net = LSTMEncoder(
            input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, dropout=0.1,
        )
    elif tab_enc == 'cnn':
        tab_net = CNNSeqEncoder(
            in_channels=1, hidden_size_half=hidden_size // 2, num_layers=n_layers,
        )
    elif tab_enc == 'resnet':
        tab_net = ResNetSeqEncoder(
            in_channels=1, base_filters=hidden_size * 2, n_block=n_layers, hidden_size=hidden_size,
        )
    if dataset_name == 'lifesnaps':
        in_channels = 5
    elif dataset_name == 'pmdata':
        in_channels = 4
    if mode == 'clip':
        mgc_model = CLIP(
            text_encoder=text_encoder,
            seq_net=seq_net,
            tab_net=tab_net,
            seq_emb_dim=hidden_size // 2,
        )
    elif mode == 'simclr':
        mgc_model = SimCLR(
            in_channel=in_channels, 
            seq_encoder=seq_net,
            seq_emb_dim=hidden_size // 2,
        )
    elif mode == 'byol':
        mgc_model = BYOL(
            in_channel=in_channels, 
            seq_encoder=seq_net,
        )
    elif mode == 'leaves':
        mgc_model = SimCLR(
            in_channel=in_channels, 
            seq_encoder=seq_net,
            use_leaves=True,
        )
    elif mode == 'sup':
        mgc_model = SupBaselineNet(
            seq_net=seq_net,
            tab_net=tab_net,
            seq_emb_dim=hidden_size // 2,
        )
    return mgc_model
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pmdata')
    parser.add_argument('--exp', type=str, default='test')
    parser.add_argument('--mode', type=str, default='clip')
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--seq_enc', type=str, default='lstm')
    parser.add_argument('--tab_enc', type=str, default='transformer')
    parser.add_argument('--save_weights', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--exclude', type=str, default='none')
    parser.add_argument('--unlabel_ratio', type=float, default=0.0)
    parser.add_argument('--label_ratio', type=float, default=1.0)
    parser.add_argument('--norm_config', type=str, default='none')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    dataset_name = args.dataset
    mode = args.mode
    n_epochs = args.n_epochs
    lr = args.lr
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    n_layers = args.n_layers
    fold = args.fold
    norm_config = args.norm_config
    seed = args.seed
    labeled_ratio = args.label_ratio
    
    # random seed
    torch.manual_seed(seed)
    
    if norm_config == 'none':
        scaler = 'none'
        norm_type = 'none'
    else:
        scaler, norm_type = norm_config.split('_')
    
    assert dataset_name in ['lifesnaps', 'pmdata']
    
    seq_enc = args.seq_enc
    tab_enc = args.tab_enc
    
    mgc_model = get_model(
        seq_enc=seq_enc,
        tab_enc=tab_enc,
        dataset_name=dataset_name,
        mode=mode,
        hidden_size=hidden_size,
        n_layers=n_layers,
    )
    
    # train the model
    if dataset_name == 'lifesnaps':
        train_loader = get_lifesnaps_loader(
            'train', batch_size=batch_size, exclude=args.exclude,
            fold=fold, unlabeled_proportion=args.unlabel_ratio, 
            labeled_proportion=labeled_ratio,
            scaler=scaler, norm_type=norm_type,
        )
        eval_loader = get_lifesnaps_loader(
            'test', batch_size=batch_size, exclude=args.exclude,
            fold=fold, unlabeled_proportion=args.unlabel_ratio, 
            labeled_proportion=labeled_ratio,
            scaler=scaler, norm_type=norm_type,
        )
    elif dataset_name == 'pmdata':
        train_loader = get_pmdata_loader(
            'train', batch_size=batch_size, exclude=args.exclude, fold=fold,
            unlabeled_proportion=args.unlabel_ratio, 
            labeled_proportion=labeled_ratio,
            scaler=scaler, norm_type=norm_type,
        )
        eval_loader = get_pmdata_loader(
            'test', batch_size=batch_size, exclude=args.exclude,
            fold=fold, unlabeled_proportion=args.unlabel_ratio, 
            labeled_proportion=labeled_ratio,
            scaler=scaler, norm_type=norm_type,
        )
    
    # training
    def train(save_path=None):
        save_weights = args.save_weights
        if args.train:
            best_model = train_loop(
                train_loader,
                eval_loader,
                mgc_model,
                n_epochs=n_epochs,
                lr=lr,
                dataset_name=dataset_name,
                mode=mode,
                save_weights=save_weights,
                save_path=save_path,
            )
        if mode in ['clip', 'simclr', 'byol', 'leaves']:
            if args.linear:
                finetune_clip_loop(
                    train_loader,
                    eval_loader,
                    best_model,
                    n_epochs=n_epochs,
                    lr=lr,
                    dataset_name=dataset_name,
                    save_weights=save_weights,
                    freeze=True,
                    save_path=save_path,
                )
            if args.finetune:
                finetune_clip_loop(
                    train_loader,
                    eval_loader,
                    best_model,
                    n_epochs=n_epochs,
                    lr=lr,
                    dataset_name=dataset_name,
                    save_weights=save_weights,
                    freeze=False,
                    save_path=save_path,
                )

    # save args as a json file
    args_dict = vars(args)
    args_dict.pop('norm_config')
    args_dict['scaler'] = scaler
    args_dict['norm_type'] = norm_type
    if scaler == 'none' and norm_type == 'none':
        norm_str = 'unnormalized'
    else:
        norm_str = f'{scaler}_{norm_type}'
    run_name = f'{dataset_name}/{args.exp}/{mode}/{norm_str}/seq:{args.seq_enc}_tab:{args.tab_enc}/'
    exp_name = f'{run_name}/unlabel_rate_{int(args.unlabel_ratio * 100)}/'
    exp_name += f'label_rate_{int(args.label_ratio * 100)}/'
    fold_name = f'{args.exclude}/fold_{fold}/seed_{seed}/'
    exp_name += fold_name
    save_path = os.path.join(EXPS_PATH, exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    time_start = time.time()
    print('Running:')
    print(json.dumps(args_dict, indent=4))
    print('-' * 80)
    train(save_path=save_path)
    print(f'Time taken: {time.time() - time_start:.4f} seconds')
    print()

if __name__ == '__main__':
    main()
