from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from mydataset import SentPairDataset
from loss_fn import CLUB_NCE, CLUB, InfoNCE, manifold_consistency_loss

def compute_loss(raw_tensors, idxs, pos_emb, neg_emb, clubnce_est, club, infonce, lambda_lower, lambda_upper, temperature):
    """
    emb_pos: [B, hidden_size]
    emb_neg: [B, hidden_size]
    """
    raw_embedding = raw_tensors[idxs]
    
    pos_mc_loss = manifold_consistency_loss(raw_embedding, pos_emb)
    neg_mc_loss = manifold_consistency_loss(raw_embedding, neg_emb)

    lower_loss = infonce(pos_emb, neg_emb)

    loss = -lambda_lower * lower_loss + pos_mc_loss + lambda_lower * neg_mc_loss
    return loss

def train(
    raw_tensors,
    file_path: str = './FineTune/data/data.json',
    model_name: str = "your best model path",
    output_dir: str = "./FineTune/output/fine_tune_model",
    batch_size: int = 256,
    epochs: int = 50,
    early_stop: int = 5,
    lr: float = 2e-4,
    max_length: int = 100,
    temperature: float = 0.07,
    lambda_lower: float = 1,
    lambda_upper: float = 0.5,
    grad_clip: float = 1.0,
):
    os.makedirs(output_dir, exist_ok=True)

    # 保存训练配置
    json.dump({
        "model_name": model_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "max_length": max_length,
        "temperature": temperature,
        "lambda_lower": lambda_lower,
        "lambda_upper": lambda_upper
    }, open(os.path.join(output_dir, "myconfig.json"), "w"), indent=2)

    full_dataset = SentPairDataset(file_path, tokenizer_name=model_name, max_length=max_length)
    train_indices, valid_indices = train_test_split(list(range(len(full_dataset))), test_size=0.1, random_state=42)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(full_dataset, valid_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    config = AutoConfig.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name, config=config)
    emb_dim = base_model.config.hidden_size
    clubnce_est = CLUB_NCE(emb_dim=emb_dim)
    club = CLUB(emb_dim=emb_dim)
    infonce = InfoNCE(emb_dim=emb_dim)

    main_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model = torch.nn.DataParallel(base_model).to(main_device)
    clubnce_est = clubnce_est.to(main_device)
    club = club.to(main_device)
    infonce = infonce.to(main_device)

    optimizer = AdamW(
        list(filter(lambda p: p.requires_grad, base_model.parameters())) +
        list(club.parameters()) +
        list(infonce.parameters()),
        lr=lr
    )

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    best_val_loss = float("inf")
    cnt = 0

    raw_tensors = raw_tensors.to(main_device)

    for epoch in range(epochs):
        base_model.train()
        clubnce_est.train()
        club.train()
        infonce.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            pos_ids = batch["pos_input_ids"].to(main_device)
            pos_mask = batch["pos_attention_mask"].to(main_device)
            neg_ids = batch["neg_input_ids"].to(main_device)
            neg_mask = batch["neg_attention_mask"].to(main_device)
            idxs = batch["idx"].to(main_device)

            hidden_pos = base_model(input_ids=pos_ids, attention_mask=pos_mask, return_dict=True).last_hidden_state
            mask_pos_exp = pos_mask.unsqueeze(-1).expand(hidden_pos.size())
            avg_pos = (hidden_pos * mask_pos_exp).sum(dim=1) / mask_pos_exp.sum(dim=1).clamp(min=1e-9)

            hidden_neg = base_model(input_ids=neg_ids, attention_mask=neg_mask, return_dict=True).last_hidden_state
            mask_neg_exp = neg_mask.unsqueeze(-1).expand(hidden_neg.size())
            avg_neg = (hidden_neg * mask_neg_exp).sum(dim=1) / mask_neg_exp.sum(dim=1).clamp(min=1e-9)

            loss = compute_loss(raw_tensors=raw_tensors, idxs=idxs, pos_emb=avg_pos, neg_emb=avg_neg,
                                clubnce_est=clubnce_est, 
                                club=club, infonce=infonce, lambda_lower=lambda_lower,
                                lambda_upper=lambda_upper, temperature=temperature)

            optimizer.zero_grad()
            # loss = loss.mean()
            loss.backward()
            clip_grad_norm_(base_model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix(train_loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        base_model.eval()
        clubnce_est.eval()
        club.eval()
        infonce.eval()
        val_loss = 0.0

        with torch.no_grad():
            vbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
            for batch in vbar:
                pos_ids = batch["pos_input_ids"].to(main_device)
                pos_mask = batch["pos_attention_mask"].to(main_device)
                neg_ids = batch["neg_input_ids"].to(main_device)
                neg_mask = batch["neg_attention_mask"].to(main_device)
                idxs = batch["idx"].to(main_device)

                hidden_pos = base_model(input_ids=pos_ids, attention_mask=pos_mask, return_dict=True).last_hidden_state
                mask_pos_exp = pos_mask.unsqueeze(-1).expand(hidden_pos.size())
                avg_pos = (hidden_pos * mask_pos_exp).sum(dim=1) / mask_pos_exp.sum(dim=1).clamp(min=1e-9)

                hidden_neg = base_model(input_ids=neg_ids, attention_mask=neg_mask, return_dict=True).last_hidden_state
                mask_neg_exp = neg_mask.unsqueeze(-1).expand(hidden_neg.size())
                avg_neg = (hidden_neg * mask_neg_exp).sum(dim=1) / mask_neg_exp.sum(dim=1).clamp(min=1e-9)

                loss = compute_loss(raw_tensors=raw_tensors, idxs=idxs, pos_emb=avg_pos, neg_emb=avg_neg,
                                    clubnce_est=clubnce_est, 
                                    club=club, infonce=infonce,
                                    lambda_lower=lambda_lower, lambda_upper=lambda_upper, temperature=temperature)
                val_loss += loss.item()
                vbar.set_postfix(valid_loss=loss.item())

        avg_val_loss = val_loss / len(valid_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            base_model.module.save_pretrained(os.path.join(output_dir, "best_model"))
            torch.save(club.state_dict(), os.path.join(output_dir, "CLUB.pt"))
            torch.save(infonce.state_dict(), os.path.join(output_dir, "InfoNCE.pt"))
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            print(f">>> Best model saved at epoch {epoch+1}")
            cnt = 0
        else:
            cnt += 1
            if cnt >= early_stop:
                print(f">>> Early stopping at epoch {epoch+1}")
                break

        base_model.module.save_pretrained(os.path.join(output_dir, f"model_epoch_{epoch+1}"))

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    raw_tensors = torch.load("./FineTune/data/text_feature/tensor.pt")
    train(raw_tensors)
