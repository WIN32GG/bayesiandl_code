#!/bin/python3
from dpt.helper import DVICPipelineWrapper, DVICContainerOperation, noop
from argparse import ArgumentParser
from dataclasses import dataclass
import uuid


BASE_PATH               = "/data" # Base persistent path in container
MOUNT_PATH              = "/data/dl/Bayeformers" # On host (poor mans pvc)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--delta', type=float, default=1e-3)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--skip_n_firsts', type=int, default=216)
    parser.add_argument('--base_epochs', type=int, default=3)
    parser.add_argument('--baye_epochs', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=64*12)
    parser.add_argument('--model_name', type=str, default="bert-base-uncased")
    parser.add_argument('--num_labels', type=int, default=384)
    parser.add_argument('--lr', type=float, default=1e-3)


    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--alpha', type=float, default=2.)
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--gamma', type=float, default=3.)

    parser.add_argument('--auto_name', action='store_true')
    parser.add_argument('--name', type=str, default="unnamed") #squad_1e3delta_calibrated_256_bert_BAM_low_labels_1epoch_scalers_3sampl2

    parser.add_argument('--dryrun', action="store_true")

    ns = parser.parse_args()


    EXP_LABEL               = str(uuid.uuid4()) if ns.auto_name else ns.name
    TENSORBOARD_PATH        = f"/data/dl/tensorboard/Bayeformers/{EXP_LABEL}"

    PIPELINE_NAME           = "Bayeformers"
    RUN_NAME                = f'{PIPELINE_NAME} {EXP_LABEL}'
    PIPELINE_DESC           = "Bayeforemers Experiments"
    NAMESPACE               = "dvic-kf"

    MODELS_PATH             = f'{BASE_PATH}/{EXP_LABEL}'
    EVAL_PATH               = f'{BASE_PATH}/{EXP_LABEL}'
    EXP_ID                  = str(uuid.uuid4()).replace('-', '_')
    EXP_PATH                = f'{BASE_PATH}/exps/{EXP_ID}'

    BASE_MODEL_PATH         = f'{MODELS_PATH}/base_model.pth'
    FREQ2_MODEL_PATH        = f'{MODELS_PATH}/freq2_model.pth'
    BAYE_MODEL_UNTRAINED    = f'{MODELS_PATH}/b_model_untrained.pth'
    BAYE_MODEL_TRAINED      = f'{MODELS_PATH}/b_model_trained.pth'

    FREQ_EVAL               = f'{EVAL_PATH}/freq.dup'
    FREQ2_EVAL              = f'{EVAL_PATH}/freq2.dup'
    BAYE_UNTRAINED_EVAL     = f'{EVAL_PATH}/baye_untrained.dup'
    BAYE_TRAINED_EVAL       = f'{EVAL_PATH}/baye_trained.dup'

    # QA Eval
    NQA_FREQ_EVAL               = f'{EVAL_PATH}/nqa_freq.dup'
    NQA_FREQ2_EVAL              = f'{EVAL_PATH}/nqa_freq2.dup'
    NQA_BAYE_UNTRAINED_EVAL     = f'{EVAL_PATH}/nqa_baye_untrained.dup'
    NQA_BAYE_TRAINED_EVAL       = f'{EVAL_PATH}/nqa_baye_trained.dup'

    SQUAD_RAW_PATH          = f'{BASE_PATH}/raw_datasets/squadv1'
    SQUAD_PROCESSED_PATH    = f'{BASE_PATH}/processed_datasets/squadv1'

    NQA_RAW_PATH            = f'{BASE_PATH}/raw_datasets/newsqa'
    NQA_PROCESSED_PATH      = f'{BASE_PATH}/processed_datasets/newsqa'

    with open('exp.log', 'a') as lg:
        lg.write(EXP_LABEL + ' ')
        lg.write(str(ns) + '\n')

    #Params
    NGPU                    = 1                 # How many GPU
    DELTA                   = ns.delta          #1e-3      # DELTA for Freq to baye conversion
    SAMPLES                 = ns.samples        #20        # Monte Carlo Samples for each Baye step
    FREEZE                  = ns.freeze         #False     # Freeze mu(s) in bayesian networks
    SKIP_N_FIRSTS           = ns.skip_n_firsts  #216         # 86=distilbert 216=bert-base
    BASE_EPOCHS             = ns.base_epochs    #3
    BAYE_EPOCHS             = ns.baye_epochs    #1
    HIDDEN_SIZE             = ns.hidden_size    #12 * 64   # HIDDEN_SIZE must be a multiple of ATTENTION_HEAD
    MODEL_NAME              = ns.model_name     #"bert-base-uncased"
    NUM_LABELS              = ns.num_labels     #384
    LR                      = ns.lr

    # Loss scalers
    ALPHA                   = ns.alpha #2.0 # ELBO
    BETA                    = ns.beta #1.0 # AVUC Scaller
    GAMMA                   = ns.gamma #3.0 # NLL

    MAX_GRAD_NORM           = 100

    BATCH_SIZE              = ns.batch_size #3 # * NGPU # Calibrated for DGX
    DEVICE_LIST             = f'--device_list 0,{NGPU}'
    SKIP_IF_EXISTS          = '-s True'

    NUM_WORKERS             = 0

    # helper

    if FREEZE:
        FREEZE = "--freeze "
    else:
        FREEZE = " "

    @dataclass
    class GraphInputData:
        label: str
        dumper_path: str
        is_baye: bool = False

        @property
        def baye(self):
            return 'b' if self.is_baye else 'f'

    def evaluator(model_path, dataset_path, eval_output, is_bayesian, name):
        baye_arg = '--type bayesian' if is_bayesian else '--type frequentist' 
        return DVICContainerOperation("win32gg/squad_multitool", "evaluate.py",
            *f'-l {model_path} --skip_n_firsts {SKIP_N_FIRSTS} -d {dataset_path} -o {eval_output} {baye_arg} {SKIP_IF_EXISTS} --hidden_size {HIDDEN_SIZE} --samples {SAMPLES} --model_name {MODEL_NAME} {DEVICE_LIST} --batch_size {BATCH_SIZE}'.split(' '), 
            name = name
        ).select_node().mount_host_path("/data", MOUNT_PATH).gpu(NGPU)

    def graph(dataset_path: str, graph_type: str, name: str, *data_sources: GraphInputData, bayesian_uncertainty_aggregator='distrib_std'):
        #graph.py -d ../dataset/processed/squadv1 -o graph.jpg -g uncertainty_repartition -s ./dump.freq.dump,Frequentist Baseline,f -s ./dump.baye_untrained.dump,Bayesian Baseline,b
        sources_args = '-s ' + ' -s '.join([f'{d.dumper_path},{d.label},{d.baye}' for d in data_sources])
        return DVICContainerOperation("win32gg/squad_multitool", "graph.py",
            *f'-g {graph_type} -d {dataset_path} --bayesian_uncertainty_aggregator {bayesian_uncertainty_aggregator} -o /graph.jpg {sources_args}'.split(' '), 
            name = name
        ).select_node().mount_host_path("/data", MOUNT_PATH).file_output('graph', '/graph.jpg')

    with DVICPipelineWrapper(PIPELINE_NAME, PIPELINE_DESC, RUN_NAME, EXP_ID, NAMESPACE) as pipeline:

        # - Data Preparation
        # Dataset
        squad_processor             = DVICContainerOperation("win32gg/squad_processor", 
                                        *f'-d {SQUAD_RAW_PATH} -o {SQUAD_PROCESSED_PATH} --train --test --fake'.split(' '),
                                        name = "squad_dataset_processor"
                                    ).select_node().mount_host_path("/data", MOUNT_PATH)

        news_qa_processor           = DVICContainerOperation("win32gg/squad_processor", 
                                        *f'-d {NQA_RAW_PATH} -o {NQA_PROCESSED_PATH} --test --fake'.split(' '),
                                        name = "nqa_dataset_processor"
                                    ).select_node().mount_host_path("/data", MOUNT_PATH)
    
        # Training
        # Base frequentist training
        frequentist_base_train      = DVICContainerOperation("win32gg/squad_multitool", "train.py",
                                        *f'-d {SQUAD_PROCESSED_PATH} --num_workers {NUM_WORKERS} -o {BASE_MODEL_PATH} --hidden_size {HIDDEN_SIZE} --lr {LR} --epochs {BASE_EPOCHS} {SKIP_IF_EXISTS} {DEVICE_LIST} --model_name {MODEL_NAME} --batch_size {BATCH_SIZE} --num_labels {NUM_LABELS}'.split(' '),
                                        name = "base_train"
                                    ).select_node().mount_host_path("/data", MOUNT_PATH).gpu(NGPU)

        frequentist_second_train    = DVICContainerOperation("win32gg/squad_multitool", "train.py",
                                        *f'-d {SQUAD_PROCESSED_PATH} -l {BASE_MODEL_PATH} --num_workers {NUM_WORKERS} --hidden_size {HIDDEN_SIZE} --lr {LR}  --epochs {BAYE_EPOCHS} -o {FREQ2_MODEL_PATH} {SKIP_IF_EXISTS} {DEVICE_LIST} --model_name {MODEL_NAME} --batch_size {BATCH_SIZE} --num_labels {NUM_LABELS}'.split(' '),
                                        name = "freq_second_train"
                                    ).select_node().mount_host_path("/data", MOUNT_PATH).gpu(NGPU)

        bayesian_save               = DVICContainerOperation("win32gg/squad_multitool", "train.py",
                                        *f'-d {SQUAD_PROCESSED_PATH} -l {BASE_MODEL_PATH} --num_workers {NUM_WORKERS} -o {BAYE_MODEL_UNTRAINED} --lr {LR}  --hidden_size {HIDDEN_SIZE} --delta {str(DELTA)} --skip_n_firsts {SKIP_N_FIRSTS} --model_name {MODEL_NAME} --type bayesian -t True --epochs 0 {SKIP_IF_EXISTS} {DEVICE_LIST}'.split(' '),
                                        name = "convert_model"
                                    ).select_node().mount_host_path("/data", MOUNT_PATH).gpu(NGPU)

        bayesian_training           = DVICContainerOperation("win32gg/squad_multitool", "train.py",
                                        *f'-d {SQUAD_PROCESSED_PATH} --num_workers {NUM_WORKERS} --beta {BETA} --gamma {GAMMA} --alpha {ALPHA} --lr {LR}  --hidden_size {HIDDEN_SIZE} --max_grad_norm {MAX_GRAD_NORM} -l {BAYE_MODEL_UNTRAINED} --model_name {MODEL_NAME} --skip_n_firsts {SKIP_N_FIRSTS} --delta {str(DELTA)} --epochs {BAYE_EPOCHS} -o {BAYE_MODEL_TRAINED} --type bayesian {SKIP_IF_EXISTS} {DEVICE_LIST} --batch_size {BATCH_SIZE} --num_labels {NUM_LABELS}'.split(' '),
                                        name = "bayesian_train"
                                    ).select_node().mount_host_path("/data", MOUNT_PATH).gpu(NGPU).tensorboard(TENSORBOARD_PATH)

        # Evaluations
        frequentist_evaluation          = evaluator(BASE_MODEL_PATH,        SQUAD_PROCESSED_PATH, FREQ_EVAL,            False, "freq_eval") 
        frequentist2_evaluation         = evaluator(FREQ2_MODEL_PATH,       SQUAD_PROCESSED_PATH, FREQ2_EVAL,           False, "freq2_eval") 
        bayesian_untrained_evaluation   = evaluator(BAYE_MODEL_UNTRAINED,   SQUAD_PROCESSED_PATH, BAYE_UNTRAINED_EVAL,  True,  "baye_untrained_eval") 
        bayesian_trained_evaluation     = evaluator(BAYE_MODEL_TRAINED,     SQUAD_PROCESSED_PATH, BAYE_TRAINED_EVAL,    True,  "baye_trained_eval")     

        # NQA Evaluations
        frequentist_evaluation_nqa          = evaluator(BASE_MODEL_PATH,        NQA_PROCESSED_PATH, NQA_FREQ_EVAL,            False, "freq_eval_nqa") 
        frequentist2_evaluation_nqa         = evaluator(FREQ2_MODEL_PATH,       NQA_PROCESSED_PATH, NQA_FREQ2_EVAL,           False, "freq2_eval_nqa") 
        bayesian_untrained_evaluation_nqa   = evaluator(BAYE_MODEL_UNTRAINED,   NQA_PROCESSED_PATH, NQA_BAYE_UNTRAINED_EVAL,  True,  "baye_untrained_eval_nqa") 
        bayesian_trained_evaluation_nqa     = evaluator(BAYE_MODEL_TRAINED,     NQA_PROCESSED_PATH, NQA_BAYE_TRAINED_EVAL,    True,  "baye_trained_eval_nqa")     


        # Execution order
        squad_processor | frequentist_base_train | frequentist_second_train
        frequentist_base_train | bayesian_save | bayesian_training

        #  wait for all evals before graphs
        wait_all_evals = noop("wait_all_evals")

        #  Evals
        bayesian_save               | bayesian_untrained_evaluation | wait_all_evals
        frequentist_base_train      | frequentist_evaluation        | wait_all_evals
        bayesian_training           | bayesian_trained_evaluation   | wait_all_evals
        frequentist_second_train    | frequentist2_evaluation       | wait_all_evals

        bayesian_save               | bayesian_untrained_evaluation_nqa | wait_all_evals
        frequentist_base_train      | frequentist_evaluation_nqa        | wait_all_evals
        bayesian_training           | bayesian_trained_evaluation_nqa   | wait_all_evals
        frequentist_second_train    | frequentist2_evaluation_nqa       | wait_all_evals

        news_qa_processor | bayesian_untrained_evaluation_nqa   | wait_all_evals
        news_qa_processor | frequentist_evaluation_nqa          | wait_all_evals
        news_qa_processor | bayesian_trained_evaluation_nqa     | wait_all_evals
        news_qa_processor | frequentist2_evaluation_nqa         | wait_all_evals

        # Graphs
        
        graph_data = [
            GraphInputData('Frequentist base',    FREQ_EVAL),
            GraphInputData('Frequentist trained', FREQ2_EVAL),
            GraphInputData('Bayesian untrained',  BAYE_UNTRAINED_EVAL, True),
            GraphInputData('Bayesian trained',    BAYE_TRAINED_EVAL,   True)
        ]
        
        wait_all_evals | graph(SQUAD_PROCESSED_PATH, 'precision_with_uncertainty', 'graph_precision',              *graph_data)
        wait_all_evals | graph(SQUAD_PROCESSED_PATH, 'uncertainty_with_precision', 'graph_uncertainty_predictive', *graph_data, bayesian_uncertainty_aggregator='predictive_uncertainty') 
        
        wait_all_evals | graph(SQUAD_PROCESSED_PATH, 'expected_calibration_error', 'ece_graph',                    *graph_data)
        wait_all_evals | graph(SQUAD_PROCESSED_PATH, 'expected_calibration_error', 'ece_graph_ent_mean',           *graph_data, bayesian_uncertainty_aggregator='entropy_mean')
        wait_all_evals | graph(SQUAD_PROCESSED_PATH, 'expected_calibration_error', 'ece_graph_pu',                 *graph_data, bayesian_uncertainty_aggregator='predictive_uncertainty')
        

        # Graphs qa
        graph_data_qa = [
            GraphInputData('Frequentist base NQA',    NQA_FREQ_EVAL),
            GraphInputData('Frequentist trained NQA', NQA_FREQ2_EVAL),
            GraphInputData('Bayesian untrained NQA',  NQA_BAYE_UNTRAINED_EVAL, True),
            GraphInputData('Bayesian trained NQA',    NQA_BAYE_TRAINED_EVAL,   True)
        ]

        wait_all_evals | graph(NQA_PROCESSED_PATH, 'precision_with_uncertainty', 'graph_precision_nqa',              *graph_data_qa)
        wait_all_evals | graph(NQA_PROCESSED_PATH, 'uncertainty_with_precision', 'graph_uncertainty_predictive_nqa', *graph_data_qa, bayesian_uncertainty_aggregator='predictive_uncertainty') 
        
        wait_all_evals | graph(NQA_PROCESSED_PATH, 'expected_calibration_error', 'ece_graph_nqa',                     *graph_data_qa)
        wait_all_evals | graph(NQA_PROCESSED_PATH, 'expected_calibration_error', 'ece_graph_nqa_ent_mean',            *graph_data_qa, bayesian_uncertainty_aggregator='entropy_mean')
        wait_all_evals | graph(NQA_PROCESSED_PATH, 'expected_calibration_error', 'ece_graph_nqa_pu',                  *graph_data_qa, bayesian_uncertainty_aggregator='predictive_uncertainty')

        # Start pipeline
        if not ns.dryrun:
            pipeline()

    """
    #Params
    NGPU                    = 1         # How many GPU
    DELTA                   = 1e-3      # DELTA for Freq to baye conversion
    SAMPLES                 = 20        # Monte Carlo Samples for each Baye step
    FREEZE                  = False     # Freeze mu(s) in bayesian networks
    SKIP_N_FIRSTS           = 216       # 86=distilbert 216=bert-base
    BASE_EPOCHS             = 3
    BAYE_EPOCHS             = 1
    HIDDEN_SIZE             = 12 * 64   # HIDDEN_SIZE must be a multiple of ATTENTION_HEAD
    MODEL_NAME              = "bert-base-uncased"
    NUM_LABELS              = 384

    # Loss scalers
    ALPHA                   = 2.0 # ELBO
    BETA                    = 1.0 # AVUC Scaller
    GAMMA                   = 3.0 # NLL
        """