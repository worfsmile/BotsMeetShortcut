
import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.nn import global_mean_pool

@torch.no_grad()
def eval_model(model, device, loader, evaluator, eval_metric='acc', save_pred=False):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            is_labeled = batch.y == batch.y
            if eval_metric == 'acc':
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'rocauc':
                pred = F.softmax(pred, dim=-1)[:, 1].unsqueeze(-1)
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(pred.detach().view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.unsqueeze(-1).detach().cpu())
            elif eval_metric == 'mat':
                y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'ap':
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
            else:
                if is_labeled.size() != pred.size():
                    with torch.no_grad():
                        pred, rep = model(batch, return_data="rep", debug=True)
                        print(rep.size())
                    print(batch)
                    print(global_mean_pool(batch.x, batch.batch).size())
                    print(pred.shape)
                    print(batch.y.size())
                    print(sum(is_labeled))
                    print(batch.y)
                batch.y = batch.y[is_labeled]
                pred = pred[is_labeled]
                y_true.append(batch.y.view(pred.shape).unsqueeze(-1).detach().cpu())
                y_pred.append(pred.detach().unsqueeze(-1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    if eval_metric == 'mat':
        res_metric = matthews_corrcoef(y_true, y_pred)
    else:
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        res_metric = evaluator.eval(input_dict)[eval_metric]
        
    # accuracy = accuracy_score(y_true, (y_pred)>0.5)
    # precision = precision_score(y_true, y_pred, average='binary')
    # recall = recall_score(y_true, y_pred, average='binary')
    # f1 = f1_score(y_true, y_pred, average='binary')
    # # Print the results
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    
    if save_pred:
        return res_metric, y_pred
    else:
        return res_metric

# import torch
# import torch.nn.functional as F
# from sklearn.metrics import matthews_corrcoef
# from torch_geometric.nn import global_mean_pool

# @torch.no_grad()
# def eval_model(model, device, loader, evaluator, eval_metric='acc', save_pred=False, verbose=False):
#     model.eval()
#     y_true = []
#     y_pred = []

#     for batch_id, batch in enumerate(loader):
#         batch = batch.to(device)

#         # Skip if only 1 node
#         if batch.x.shape[0] == 1:
#             if verbose:
#                 print(f"Skipping batch {batch_id} due to single node.")
#             continue

#         with torch.no_grad():
#             output = model(batch)

#         # Label mask
#         is_labeled = batch.y == batch.y  # filter NaNs

#         # Skip if no labeled data
#         if is_labeled.sum() == 0:
#             if verbose:
#                 print(f"Skipping batch {batch_id} due to no labeled nodes.")
#             continue

#         pred = output

#         # Process based on metric
#         if eval_metric == 'acc':
#             y_true.append(batch.y.view(-1, 1).detach().cpu())
#             y_pred.append(pred.argmax(dim=1).view(-1, 1).detach().cpu())

#         elif eval_metric == 'rocauc':
#             prob = F.softmax(pred, dim=-1)[:, 1].unsqueeze(-1)
#             y_true.append(batch.y.view(-1, 1).detach().cpu())
#             y_pred.append(prob.detach().cpu())

#         elif eval_metric == 'mat':
#             y_true.append(batch.y.view(-1, 1).detach().cpu())
#             y_pred.append(pred.argmax(dim=-1).view(-1, 1).detach().cpu())

#         elif eval_metric == 'ap':
#             if batch.y.numel() != pred.numel():
#                 if verbose:
#                     print(f"Warning: batch {batch_id} y/pred shape mismatch: {batch.y.shape} vs {pred.shape}")
#                 continue
#             y_true.append(batch.y.view(pred.shape).detach().cpu())
#             y_pred.append(pred.detach().cpu())

#         else:
#             # Generic fallback
#             batch.y = batch.y[is_labeled]
#             pred = pred[is_labeled]

#             if batch.y.shape[0] != pred.shape[0]:
#                 if verbose:
#                     print(f"Mismatch in fallback metric: y={batch.y.shape}, pred={pred.shape}")
#                 continue

#             y_true.append(batch.y.view(pred.shape).detach().cpu())
#             y_pred.append(pred.detach().cpu())

#     # Check if nothing collected
#     if len(y_true) == 0 or len(y_pred) == 0:
#         print("No valid predictions made during evaluation.")
#         return 0.0

#     y_true = torch.cat(y_true, dim=0).numpy()
#     y_pred = torch.cat(y_pred, dim=0).numpy()

#     # Final shape check
#     if y_true.shape != y_pred.shape:
#         raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

#     if eval_metric == 'mat':
#         res_metric = matthews_corrcoef(y_true, y_pred)
#     else:
#         input_dict = {"y_true": y_true, "y_pred": y_pred}
#         res_metric = evaluator.eval(input_dict)[eval_metric]

#     return res_metric
