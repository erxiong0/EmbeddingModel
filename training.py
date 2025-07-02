#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
import sys
import traceback
from datetime import datetime
import os

from datasets import load_from_disk
from config import MODEL_PATH, DATA_PATH, STS_PATH, STS_B_PATH

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# 尝试导入wandb
try:
    import wandb
    WANDB_AVAILABLE = True
    logging.info("Wandb已安装，将启用追踪")
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("Wandb未安装，将跳过追踪")

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def setup_wandb():
    """设置Wandb配置"""
    if not WANDB_AVAILABLE:
        return None
    
    try:
        # 初始化wandb
        wandb.init(
            project="embedding-model-training",
            name=f"position-sts-embedding-balanced-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "model_name": MODEL_PATH.split("/")[-1],
                "train_batch_size": 128,  
                "num_train_epochs": 3,    
                "learning_rate": 3e-6,    
                "warmup_ratio": 0.1,      
                "fp16": True,
                "eval_steps": 300,        
                "save_steps": 30000,      
                "logging_steps": 30,      
                "gradient_accumulation_steps": 4,  
                "weight_decay": 0.01,     
                "dataset_path": DATA_PATH,
                "sts_path": STS_PATH,
                "sts_b_path": STS_B_PATH
            },
            tags=["embedding", "sentence-transformers", "position-matching", "balanced"]
        )
        
        logging.info(f"Wandb项目: {wandb.run.project}")
        logging.info(f"Wandb运行ID: {wandb.run.id}")
        
        return wandb.run
        
    except Exception as e:
        logging.error(f"设置Wandb失败: {e}")
        return None

def main():
    """主函数"""
    wandb_run = setup_wandb()
    
    try:
        model_name = MODEL_PATH.split("/")[-1]
        train_batch_size = 128  
        
        output_dir = "output/training_nli_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        logging.info(f"加载模型: {MODEL_PATH}")
        model = SentenceTransformer(MODEL_PATH)
        
        if wandb_run:
            wandb.config.update({
                "model_embedding_dimension": model.get_sentence_embedding_dimension(),
                "model_max_seq_length": model.max_seq_length
            })
        
        logging.info("加载数据集...")
        train_dataset = load_from_disk(DATA_PATH)['train'].select(range(150000))  # 减少训练数据
        eval_dataset = load_from_disk(DATA_PATH)['eval'].select(range(30000))     # 减少评估数据
        
        if wandb_run:
            wandb.config.update({
                "train_dataset_size": len(train_dataset),
                "eval_dataset_size": len(eval_dataset),
                "train_dataset_columns": train_dataset.column_names
            })
        
        logging.info(f"训练数据集: {len(train_dataset)} 样本")
        logging.info(f"评估数据集: {len(eval_dataset)} 样本")
        
        train_loss = losses.SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=2,
        )
        
        logging.info("设置评估器...")
        stsb_eval_dataset = load_from_disk(STS_PATH)
        dev_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=stsb_eval_dataset["sentence1"],
            sentences2=stsb_eval_dataset["sentence2"],
            scores=stsb_eval_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name="sts-dev",
        )
        
        if wandb_run:
            wandb.config.update({
                "sts_eval_size": len(stsb_eval_dataset)
            })
        
        logging.info("评估器设置完成")
        

        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,                     
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=train_batch_size,
            gradient_accumulation_steps=4,          
            warmup_ratio=0.1,                       
            learning_rate=3e-6,                     
            weight_decay=0.01,                      
            fp16=True,
            bf16=False,
            eval_strategy="steps",
            eval_steps=300,                         
            save_strategy="steps",
            save_steps=30000,                       
            save_total_limit=5,                     
            logging_steps=30,                       
            run_name="Position_STS_Embedding_Balanced",
            load_best_model_at_end=True,
            metric_for_best_model="eval_sts-dev_pearson_cosine",
            greater_is_better=True,
            report_to=["wandb"] if WANDB_AVAILABLE else [],
        )
        
        logging.info("开始训练...")
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=train_loss,
            evaluator=dev_evaluator,
        )
        
        logging.info("训练前评估:")
        initial_eval = dev_evaluator(model)
        if wandb_run:
            wandb.log({"initial_sts_score": initial_eval})
        
        trainer.train()
        
        logging.info("最终评估...")
        test_dataset = load_from_disk(STS_B_PATH)
        test_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=test_dataset["sentence1"],
            sentences2=test_dataset["sentence2"],
            scores=test_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name="sts-test",
        )
        
        final_eval = test_evaluator(model)
        if wandb_run:
            wandb.log({"final_sts_score": final_eval})
        
        final_output_dir = f"{output_dir}/final"
        model.save(final_output_dir)
        logging.info(f"模型已保存到: {final_output_dir}")
        
        # if wandb_run:
        #     model_artifact = wandb.Artifact(
        #         name=f"embedding-model-balanced-{wandb.run.id}",
        #         type="model",
        #         description="平衡调优的Embedding模型"
        #     )
        #     model_artifact.add_dir(final_output_dir)
        #     wandb.log_artifact(model_artifact)
        
        # model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
        # try:
        #     hub_model_name = f"{model_name}-nli-balanced"
        #     model.push_to_hub(hub_model_name)
        #     logging.info(f"模型已上传到Hugging Face Hub: {hub_model_name}")
            
        #     if wandb_run:
        #         wandb.config.update({"hub_model_name": hub_model_name})
                
        # except Exception as e:
        #     logging.warning(f"上传到Hugging Face Hub失败 (权限问题): {e}")
        #     if wandb_run:
        #         wandb.config.update({"hub_upload_failed": True, "hub_error": str(e)})
        
        # if wandb_run:
        #     wandb.finish()
        
        logging.info("平衡训练完成!")
        
    except Exception as e:
        logging.error(f"训练过程中出错: {e}")
        if wandb_run:
            wandb.finish(exit_code=1)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()