import torch
from transformers import AutoTokenizer, T5Model
import model
import utils
import myevaluate
from tqdm import tqdm

def backbone(data_name, test_rec_loader, user_emb, item_emb, item_num, args, device, pop_weights):
    linear_projection = model.projection(input_dim=512, output_dim=item_emb.shape[1], target_length=args.target_length)
    t5 = T5Model.from_pretrained('../checkpoints/backbone/' + data_name)
    tokenizer = AutoTokenizer.from_pretrained("../checkpoints/backbone/" + data_name, legacy=False)
    linear_projection.load_state_dict(torch.load('../checkpoints/backbone/' + data_name + '/projection.pt'))

    t5.to(device)
    linear_projection.to(device)
    t5.eval()
    linear_projection.eval()

    item_vq = model.DQ(input_dim=item_emb.shape[1], dim=512, n_embedding=args.n_token, m_book=args.n_book)
    try:
        item_vq.load_state_dict(torch.load('../checkpoints/vq/co-evolved-item-MQ-' + data_name + '.pth'))
    except FileNotFoundError:
        item_vq.load_state_dict(torch.load('../checkpoints/vq/item-MQ-lgn-' + data_name + '-64.pth'))
    item_vq.to(device)
    item_vq.eval()

    pop = pop_weights.to(device)
    pop_log = torch.log(pop + 1.0)
    pop_penalty = pop_log / (pop_log.max() + 1e-8)

    n_batch = 0
    metrics = torch.zeros([3, 2]).to(device)

    for sample in tqdm(test_rec_loader, desc="Testing Model"):
        _, item_id, target_id, user_cb_id, item_cb_id, target_cb_id = sample
        input_sentences = utils.prompt(user_cb_id, item_cb_id, is_test=True, is_unseen=args.is_unseen)
        input_encoding = tokenizer(input_sentences, return_tensors='pt', max_length=args.source_length,
                                   padding="max_length", truncation=True)

        decoder_input_encoding = tokenizer([args.decoder_prepend] * len(list(target_cb_id)), return_tensors="pt",
                                           max_length=args.target_length, padding="max_length", truncation=True)
        decoder_input_ids = t5._shift_right(decoder_input_encoding.input_ids)

        with torch.no_grad():
            outputs = t5(input_ids=input_encoding.input_ids.to(device),
                         attention_mask=input_encoding.attention_mask.to(device),
                         decoder_input_ids=decoder_input_ids.to(device))
            predicts = linear_projection(outputs.last_hidden_state)

            scores = utils.similarity_score(predicts, item_emb, item_id)
            fair_scores = scores - args.fair_lambda * pop_penalty.unsqueeze(0)
            results = torch.argsort(fair_scores, dim=1, descending=True)

        for j in range(metrics.shape[0]):
            metr, batch = myevaluate.get_metrics(target_id.to(device), results, device, (j + 1) * 10)
            metrics[j, :] += metr
        n_batch += batch

    print("\nCoDT Evaluation Results (@10, @20, @30):")
    for j in range(metrics.shape[0]):
        print(
            f'test_hit@{((j + 1) * 10)} = {metrics[j, 0].item() / n_batch:.4f}, test_ndcg@{((j + 1) * 10)} = {metrics[j, 1].item() / n_batch:.4f}')
