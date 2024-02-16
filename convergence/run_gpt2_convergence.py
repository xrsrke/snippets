from __future__ import print_function

import msamp
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def get_time_name():
    import datetime

    today = datetime.datetime.now()
    return today.strftime("%d/%m/%Y_%H:%M:%S")


def main():
    """The main function."""
    
    wandb.init(
        project="nanotron",
        name=f"{get_time_name()}.test_msamp_gpt2_convergence",
    )
    
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("stas/c4-en-10k")
    dataset = dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt"))
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloaders = torch.utils.data.DataLoader(dataset["train"], batch_size=10, shuffle=True)
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    bf16_model = deepcopy(model)
    
    # NOTE: this is the reference format in the paper
    bf16_model = bf16_model.half().to("cuda")
    bf16_optim = optim.Adam(bf16_model.parameters(), lr=1e-3)
    
    model = model.to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model, optimizer = msamp.initialize(model, optimizer, opt_level="O2")

    model.train()
    bf16_model.train()
    
    for _ in range(5):
        for inputs in dataloaders:
            # NOTE: move inputs to cuda
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            optimizer.zero_grad()
            bf16_optim.zero_grad()
            
            
            with torch.cuda.amp.autocast():
                output = model(**inputs)
                loss = nn.CrossEntropyLoss()(output.logits.view(-1, output.logits.shape[-1]), inputs["input_ids"].view(-1))
                loss.backward()
                optimizer.step()
            
            bf16_output = bf16_model(**inputs)
            bf16_loss = nn.CrossEntropyLoss()(bf16_output.logits.view(-1, bf16_output.logits.shape[-1]), inputs["input_ids"].view(-1))
            bf16_loss.backward()
            bf16_optim.step()
            
            wandb.log({"loss": loss.item(), "bf16_loss": bf16_loss.item()})


if __name__ == "__main__":
    main()
