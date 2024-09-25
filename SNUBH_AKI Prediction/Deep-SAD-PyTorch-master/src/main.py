import click
import torch
import logging
import random
import numpy as np
import itertools
import json
import os

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from DeepSAD import DeepSAD
from datasets.main import load_dataset

import warnings
warnings.filterwarnings("ignore")

################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'fmnist', 'cifar10', 'arrhythmia', 'cardio', 'satellite',
                                                   'satimage-2', 'shuttle', 'thyroid', 'custom']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'fmnist_LeNet', 'cifar10_LeNet', 'arrhythmia_mlp',
                                               'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                                               'thyroid_mlp','custom_mlp']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--fairness_type', type=click.Choice(['EO', 'DP']), default='EO',
              help='Type of fairness loss to use: "EO" for Equalized Odds or "DP" for Demographic Parity.')
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--eta', type=float, default=1.0, help='Deep SAD hyperparameter eta (must be 0 < eta).')
@click.option('--alpha', type=float, default=None)

@click.option('--ratio_known_normal', type=float, default=0.6,
              help='Ratio of known (labeled) normal training examples.')
@click.option('--ratio_known_outlier', type=float, default=0.4,
              help='Ratio of known (labeled) anomalous training examples.')
@click.option('--ratio_pollution', type=float, default=0.0,
              help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for Deep SAD network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SAD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=80, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=80, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--num_threads', type=int, default=0,
              help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--known_outlier_class', type=int, default=1,
              help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
@click.option('--n_known_outlier_classes', type=int, default=1,
              help='Number of known outlier classes.'
                   'If 0, no anomalies are known.'
                   'If 1, outlier class as specified in --known_outlier_class option.'
                   'If > 1, the specified number of outlier classes will be sampled at random.')
@click.option('--random_state', default=1, type=int, help='Random seed for reproducibility.')
def main(dataset_name, net_name, fairness_type, xp_path, data_path, load_config, load_model, eta,
         ratio_known_normal, ratio_known_outlier, ratio_pollution, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay,
         pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay,
         num_threads, n_jobs_dataloader, normal_class, known_outlier_class, n_known_outlier_classes,
         random_state, alpha):
    """
    Deep SAD, a method for deep semi-supervised anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = os.path.join(xp_path, 'log.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
    logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
    logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
    if n_known_outlier_classes == 1:
        logger.info('Known anomaly class: %d' % known_outlier_class)
    else:
        logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)
    logger.info('Network: %s' % net_name)

    # Prepare configuration without including logger
    if load_config:
        with open(load_config, 'r') as f:
            cfg_settings = json.load(f)
        # 'cfg' 키가 존재하면 제거
        if 'cfg' in cfg_settings:
            del cfg_settings['cfg']
        current_cfg = Config(cfg_settings)
    else:
        # 기본 설정을 사용하여 Config 객체 생성
        cfg_settings = {
            'dataset_name': dataset_name,
            'net_name': net_name,
            'fairness_type': fairness_type,
            'xp_path': xp_path,
            'data_path': data_path,
            'load_config': None,
            'load_model': None,
            'eta': eta,
            'ratio_known_normal': ratio_known_normal,
            'ratio_known_outlier': ratio_known_outlier,
            'ratio_pollution': ratio_pollution,
            'device': device,
            'seed': seed,
            'optimizer_name': optimizer_name,
            'lr': lr,
            'n_epochs': n_epochs,
            'lr_milestone': lr_milestone,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'pretrain': pretrain,
            'ae_optimizer_name': ae_optimizer_name,
            'ae_lr': ae_lr,
            'ae_n_epochs': ae_n_epochs,
            'ae_lr_milestone': ae_lr_milestone,
            'ae_batch_size': ae_batch_size,
            'ae_weight_decay': ae_weight_decay,
            'num_threads': num_threads,
            'n_jobs_dataloader': n_jobs_dataloader,
            'normal_class': normal_class,
            'known_outlier_class': known_outlier_class,
            'n_known_outlier_classes': n_known_outlier_classes,
            'alpha' : alpha
            # 'cfg' 키 제거
        }
        current_cfg = Config(cfg_settings)

    # Print model configuration
    logger.info('Eta-parameter: %.2f' % current_cfg.settings['eta'])

    # Set seed
    if current_cfg.settings['seed'] != -1:
        random.seed(current_cfg.settings['seed'])
        np.random.seed(current_cfg.settings['seed'])
        torch.manual_seed(current_cfg.settings['seed'])
        torch.cuda.manual_seed(current_cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % current_cfg.settings['seed'])

# Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

     # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                          ratio_known_normal, ratio_known_outlier, ratio_pollution,
                          random_state=random_state)
    # Log random sample of known anomaly classes if more than 1 class
    if n_known_outlier_classes > 1:
        logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

    # Initialize DeepSAD model and set neural network phi
    deepSAD = DeepSAD(current_cfg.settings['eta'], fairness_type=fairness_type)
    deepSAD.set_network(net_name)

    # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    if load_model:
        deepSAD.load_model(model_path=load_model, load_ae=True, map_location=device)
        logger.info('Loading model from %s.' % load_model)
        
    # Hyperparameter tuning 설정
    if not alpha:
        hyperparameter_grid = {
            'eta': [0.5, 1.0, 1.5],                                      # Deep SAD hyperparameter eta
            'alpha' : [round(x * 0.1, 1) for x in range(1, 11)],         # Fairness loss 가중치
            'lr': [0.001, 0.0005, 0.0001],                               # 학습률
            'batch_size': [64, 128, 256],                                # 배치 크기
            'lr_milestone': [[15, 35, 45]],                              # lr_마일스톤 (특정 epoch마다 lr 감소시키는 지점 설정)
            'weight_decay': [1e-4, 1e-5, 1e-6]                           # 가중치 감쇠
            # 필요시 다른 hyperparameter 추가
        }
    else:
        hyperparameter_grid = {
            'eta': [0.5],                                      # Deep SAD hyperparameter eta    # Fairness loss 가중치
            'lr': [0.001],                               # 학습률
            'batch_size': [64],                                # 배치 크기
            'lr_milestone': [[15, 35, 45]],                              # lr_마일스톤 (특정 epoch마다 lr 감소시키는 지점 설정)
            'weight_decay': [1e-4]                           # 가중치 감쇠
            # 필요시 다른 hyperparameter 추가
        }
    
    # Define a reasonable number of seeds to evaluate for each hyperparameter combination
    seed_list = list(range(1, 2)) 
    
    # 하이퍼파라미터 조합 생성
    keys, values = zip(*hyperparameter_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    logger.info(f'Total hyperparameter combinations to try: {len(hyperparameter_combinations)}')
    logger.info(f'Number of seeds per combination: {len(seed_list)}')

    best_auc = -1
    best_hyperparams = {}
    best_seed = None
    results_summary = []    

    for idx, hyperparams in enumerate(hyperparameter_combinations, 1):
        logger.info(f'\n=== Hyperparameter Set {idx}/{len(hyperparameter_combinations)} ===')
        logger.info(f'Hyperparameters: {hyperparams}')
        
        for seed_value in seed_list:
            logger.info(f'\n--- Seed {seed_value} ---')

            # 각 하이퍼파라미터 조합에 대해 실험을 수행하기 위해 별도의 실험 경로를 생성합니다.
            experiment_path = os.path.join(xp_path, f'experiment_{idx}_seed_{seed_value}')
            os.makedirs(experiment_path, exist_ok=True)

            # 설정을 하이퍼파라미터로 업데이트 (logger 제외)
            config_dict = {k: v for k, v in current_cfg.settings.items()}
            config_dict.update(hyperparams)
            config_dict['seed'] = seed_value
            experiment_cfg = Config(config_dict)
            
            # Set the current seed
            random.seed(seed_value)
            np.random.seed(seed_value)
            torch.manual_seed(seed_value)
            torch.cuda.manual_seed(seed_value)
            torch.backends.cudnn.deterministic = True
            logger.info('Set seed to %d.' % seed_value)    

            # 로그 파일 설정
            experiment_log_file = os.path.join(experiment_path, 'log.txt')
            # 기존 핸들러 제거 (기존의 file_handler는 계속 유지)
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler) and handler.baseFilename != log_file:
                    logger.removeHandler(handler)
                    
            # 새로운 핸들러 추가
            file_handler = logging.FileHandler(experiment_log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            logger.info(f'Starting experiment {idx} with hyperparameters: {hyperparams}')
            
            try:
                # DeepSAD 모델 초기화
                deepSAD = DeepSAD(experiment_cfg.settings['eta'], fairness_type=fairness_type)
                deepSAD.set_network(net_name)

                # 모델 로드 여부
                if load_model:
                    deepSAD.load_model(model_path=load_model, load_ae=True, map_location=device)
                    logger.info('Loading model from %s.' % load_model)

                # Pretrain
                if pretrain:
                    logger.info('Pretraining: %s' % pretrain)
                    logger.info(f'Pretraining optimizer: {experiment_cfg.settings["ae_optimizer_name"]}')
                    logger.info(f'Pretraining learning rate: {experiment_cfg.settings["ae_lr"]}')
                    logger.info(f'Pretraining epochs: {experiment_cfg.settings["ae_n_epochs"]}')
                    logger.info(f'Pretraining learning rate scheduler milestones: {experiment_cfg.settings["ae_lr_milestone"]}')
                    logger.info(f'Pretraining batch size: {experiment_cfg.settings["ae_batch_size"]}')
                    logger.info(f'Pretraining weight decay: {experiment_cfg.settings["ae_weight_decay"]}')

                    # Pretrain 모델 학습
                    deepSAD.pretrain(
                        dataset.pretrain_dataset,
                        optimizer_name=experiment_cfg.settings['ae_optimizer_name'],
                        lr=experiment_cfg.settings['ae_lr'],
                        n_epochs=experiment_cfg.settings['ae_n_epochs'],
                        lr_milestones=experiment_cfg.settings['ae_lr_milestone'],
                        batch_size=experiment_cfg.settings['ae_batch_size'],
                        weight_decay=experiment_cfg.settings['ae_weight_decay'],
                        device=device,
                        n_jobs_dataloader=n_jobs_dataloader
                    )

                    # Pretrain 결과 저장
                    deepSAD.save_ae_results(export_json=os.path.join(experiment_path, 'ae_results.json'))
                    logger.info('Pretraining completed and results saved.')

                # Log training details
                logger.info('Training optimizer: %s' % experiment_cfg.settings['optimizer_name'])
                logger.info('Training learning rate: %g' % experiment_cfg.settings['lr'])
                logger.info('Training alpha: %s' % experiment_cfg.settings['alpha'])
                logger.info('Training epochs: %d' % experiment_cfg.settings['n_epochs'])
                logger.info('Training learning rate scheduler milestones: %s' % (experiment_cfg.settings['lr_milestone'],))
                logger.info('Training batch size: %d' % experiment_cfg.settings['batch_size'])
                logger.info('Training weight decay: %g' % experiment_cfg.settings['weight_decay'])

                # Train model on dataset
                logger.info('Starting training...')
                deepSAD.train(
                    dataset,
                    alpha=experiment_cfg.settings['alpha'],
                    optimizer_name=experiment_cfg.settings['optimizer_name'],
                    lr=experiment_cfg.settings['lr'],
                    n_epochs=experiment_cfg.settings['n_epochs'],
                    lr_milestones=experiment_cfg.settings['lr_milestone'],
                    batch_size=experiment_cfg.settings['batch_size'],
                    weight_decay=experiment_cfg.settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader
                )
                logger.info('Training completed.')

                # Test model
                logger.info('Starting testing...')
                deepSAD.test(
                    dataset,
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader
                )
                logger.info('Testing completed.')
                
                # 결과, 모델, 설정 저장
                deepSAD.save_results(export_json=os.path.join(experiment_path, 'results.json'))
                deepSAD.save_model(export_model=os.path.join(experiment_path, 'model.tar'))
                experiment_cfg.save_config(export_json=os.path.join(experiment_path, 'config.json'))
                logger.info('Results, model, and config saved.')
            
               # Test AUC 및 Fair Loss 기록
                test_auc = deepSAD.results.get('test_auc', None)
                test_fair_loss = deepSAD.results.get('test_fair_loss', None)
                if test_auc is not None:
                    logger.info(f'Experiment {idx} with seed {seed_value} Test AUC: {test_auc:.2f}%')
                else:
                    logger.warning(f'Experiment {idx} with seed {seed_value} Test AUC not found.')

                if test_fair_loss is not None:
                    logger.info(f'Experiment {idx} with seed {seed_value} Test Fair Loss: {test_fair_loss:.3f}')
                else:
                    logger.warning(f'Experiment {idx} with seed {seed_value} Test Fair Loss not found.')

                
                # 최적의 하이퍼파라미터 업데이트 (Fair Loss <=0.1인 경우)
                if test_fair_loss is not None and test_fair_loss <= 0.06:
                    if test_auc > best_auc:
                        best_auc = test_auc
                        best_hyperparams = hyperparams.copy()
                        best_seed = seed_value
                        logger.info(f'New best AUC: {best_auc:.2f}%')
                        logger.info(f'seed: {best_seed}')
                        logger.info(f'Best hyperparameters: {best_hyperparams}')            
            
                # Append results
                results_summary.append({
                    'experiment_id': idx,
                    'seed': seed_value,
                    'hyperparameters': hyperparams,
                    'test_auc': test_auc,
                    'test_fair_loss': test_fair_loss
                })
        
            except Exception as e:
                logger.error(f'An error occurred during experiment {idx} with seed {seed_value}: {str(e)}')
                results_summary.append({
                    'experiment_id': idx,
                    'seed': seed_value,
                    'hyperparameters': hyperparams,
                    'test_auc': None,
                    'test_fair_loss': None,
                    'error': str(e)
                })

            finally:
                # 실험 로그 핸들러 제거
                logger.removeHandler(file_handler)
    
    # After all experiments, save the best hyperparameters
    best_hyperparams_path = os.path.join(xp_path, 'best_hyperparams.json')
    with open(best_hyperparams_path, 'w') as fp:
        json.dump({
            'best_test_auc': best_auc,
            'best_hyperparameters': best_hyperparams,
            'seed': best_seed,
        }, fp, indent=4)
    logger.info(f'\nBest Test AUC: {best_auc*100.:.2f}% | seed: {best_seed} | hyperparameters: {best_hyperparams}')
    logger.info(f'Best hyperparameters saved to {best_hyperparams_path}')

    # Optionally, save a summary of all experiments
    summary_path = os.path.join(xp_path, 'hyperparameter_search_summary.json')
    with open(summary_path, 'w') as fp:
        json.dump(results_summary, fp, indent=4)
    logger.info(f'Hyperparameter search summary saved to {summary_path}')
    
        
    # Plot most anomalous and most normal test samples
    try:
        test_scores = deepSAD.results.get('test_scores')
        if test_scores:
            indices, labels, scores = zip(*test_scores)
            indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
            idx_all_sorted = indices[np.argsort(scores)]  # from lowest to highest score
            idx_normal_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # from lowest to highest score

            if dataset_name in ('mnist', 'fmnist', 'cifar10'):

                if dataset_name in ('mnist', 'fmnist'):
                    X_all_low = dataset.test_set.data[idx_all_sorted[:32], ...].unsqueeze(1)
                    X_all_high = dataset.test_set.data[idx_all_sorted[-32:], ...].unsqueeze(1)
                    X_normal_low = dataset.test_set.data[idx_normal_sorted[:32], ...].unsqueeze(1)
                    X_normal_high = dataset.test_set.data[idx_normal_sorted[-32:], ...].unsqueeze(1)

                if dataset_name == 'cifar10':
                    X_all_low = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[:32], ...], (0,3,1,2)))
                    X_all_high = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[-32:], ...], (0,3,1,2)))
                    X_normal_low = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[:32], ...], (0,3,1,2)))
                    X_normal_high = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[-32:], ...], (0,3,1,2)))

                plot_images_grid(X_all_low, export_img=os.path.join(xp_path, 'all_low.png'), padding=2)
                plot_images_grid(X_all_high, export_img=os.path.join(xp_path, 'all_high.png'), padding=2)
                plot_images_grid(X_normal_low, export_img=os.path.join(xp_path, 'normals_low.png'), padding=2)
                plot_images_grid(X_normal_high, export_img=os.path.join(xp_path, 'normals_high.png'), padding=2)
        else:
            logger.warning('No test scores available to plot.')
    except Exception as e:
        logger.error(f'An error occurred during plotting: {str(e)}')


if __name__ == '__main__':
    main()
