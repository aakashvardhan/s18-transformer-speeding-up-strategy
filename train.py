import torch

from config_file import get_config, get_weights_file_path

torch.cuda.amp.autocast(enabled = True)

import os
import warnings
from pathlib import Path

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:12240"
config = get_config()








def run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, max_len, device, writer, global_step):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []
    
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80
        
    with torch.no_grad():
        for batch in val_dataloader:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            assert encoder_input.size(0)==1, "Batch size must be 1 for validation"
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
    """        
            print("SOURCE", source_text)
            print("TARGET", target_text)
            print("PREDICTED", model_out_text)
            
    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()
        
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()
        
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()
        
     """   




def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")
    
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    #Tensorboard
    writer = SummaryWriter(config["experiment_name"])
    
    #Adam is used to train each feature with a different learning rate. 
    #If some feature is appearing less, adam takes care of it
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], eps = 1e-9)
    
    initial_epoch = 0
    global_step = 0
    
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print("Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
        print("preloaded")
        
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1)
    
    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        print(epoch)
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"Processing Epoch {epoch:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            
            label = batch["label"].to(device)
            
            #Compute loss using cross entropy
            tgt_vocab_size = tokenizer_tgt.get_vocab_size()
            loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            #Log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()
            
            #Backpropogate loss
            loss.backward()
            
            #Update weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step+=1
            
        #run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, writer, global_step)
        
        
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step
            },
            model_filename
        )
        
            
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
    
    