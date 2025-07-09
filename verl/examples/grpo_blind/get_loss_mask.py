import transformers
import torch
import os

tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def _create_loss_mask(tokenizer, response, start_marker="<|im_start|>assistant", end_marker="<|im_end|>"):
    response_length = response.shape[-1]
    response_mask = torch.ones_like(response, dtype=torch.int8)
    loss_mask = torch.zeros_like(response_mask)
    
    # Get the token IDs for the markers
    start_marker_ids = tokenizer.encode(start_marker, add_special_tokens=False)
    end_marker_ids = tokenizer.encode(end_marker, add_special_tokens=False)
    
    # Process each sequence in the batch
    for i, response_ids in enumerate(response):
        # Convert to list for easier processing
        response_ids_list = response_ids.tolist()
        
        # Track current position in sequence
        pos = 0
        # Use a stack to track start markers for proper nesting
        marker_stack = []
        
        while pos < len(response_ids_list):
            # Look for start marker
            if _is_subsequence(response_ids_list, start_marker_ids, pos):
                # Record the position after the start marker
                marker_stack.append(pos)
                pos += len(start_marker_ids)
            # Look for end marker, but only process if we have a corresponding start marker
            elif _is_subsequence(response_ids_list, end_marker_ids, pos) and marker_stack:
                # Get the matching start position from stack
                start_pos = marker_stack.pop()
                end_pos = pos + len(end_marker_ids)  # End position is after the end marker
                
                # Apply the mask to this section - only for tokens between start and end
                loss_mask[i, start_pos:end_pos] = 1
                
                # Move past this end marker
                pos += len(end_marker_ids)
            else:
                # No markers at this position, move on
                pos += 1
    
    # Apply the response mask to ensure we only include valid tokens
    loss_mask = loss_mask * response_mask
    
    # Visualize and compare tokens with and without the loss mask
    responses_loss = []
    responses_unmasked = []
    responses_original = []

    for i, res in enumerate(response):
        masked_indices = torch.nonzero(loss_mask[i].bool() & response_mask[i].bool(), as_tuple=True)[0]
        unmasked_indices = torch.nonzero(~loss_mask[i].bool() & response_mask[i].bool(), as_tuple=True)[0]
        
        masked_tokens = res[masked_indices] if masked_indices.numel() > 0 else torch.tensor([], device=res.device, dtype=res.dtype)
        unmasked_tokens = res[unmasked_indices] if unmasked_indices.numel() > 0 else torch.tensor([], device=res.device, dtype=res.dtype)
        
        original_text = tokenizer.decode(res, skip_special_tokens=False)
        masked_text = tokenizer.decode(masked_tokens, skip_special_tokens=False)
        unmasked_text = tokenizer.decode(unmasked_tokens, skip_special_tokens=False)
        
        responses_loss.append(unmasked_text)  # Text that won't contribute to loss
        responses_unmasked.append(masked_text)  # Text that will contribute to loss
        responses_original.append(original_text)  # Original text for reference
    # print("Responses with loss mask (masked tokens):")
    # for resp in responses_loss:
    #     print(resp)
    #     print('-' * 20)

    # print("\nResponses without loss mask (unmasked tokens):")
    # for resp in responses_unmasked:
    #     print(resp)
    #     print('-' * 20)

    

    # # Debug visualization (save output files)
    dir_prefix = "/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/verl/_debug_output"
    # os.makedirs(dir_prefix, exist_ok=True)
    
    # # Save responses and loss mask
    with open(os.path.join(dir_prefix, f'debug_responses_masked_v2.txt'), 'w') as f:
        for resp in responses_loss:
            f.write(f"{resp}\n")
            f.write('\n\n' + '#' * 20 + '\n\n')  # Separator for clarity in the debug file

    with open(os.path.join(dir_prefix, f'debug_responses_unmasked_v2.txt'), 'w') as f:
        for resp in responses_unmasked:
            f.write(f"{resp}\n")
            f.write('\n\n' + '#' * 20 + '\n\n')  # Separator for clarity in the debug file

    with open(os.path.join(dir_prefix, f'debug_responses_original_v2.txt'), 'w') as f:
        for resp in responses_original:
            f.write(f"{resp}\n")
            f.write('\n\n' + '#' * 20 + '\n\n')

    # # Save batch.batch['responses']
    # torch.save(
    #     batch.batch['responses'],
    #     os.path.join(dir_prefix, f'debug_responses_tensor.pt')
    # )

    # # Save loss mask
    # torch.save(
    #     loss_mask,
    #     os.path.join(dir_prefix, f'debug_loss_mask_tensor.pt')
    # )

    # # Calculate metrics
    # total_tokens = response_mask.sum().item()
    # masked_tokens = loss_mask.sum().item()
    # metrics.update({
    #     'loss_mask/total_tokens': total_tokens,
    #     'loss_mask/masked_tokens': masked_tokens,
    #     'loss_mask/percentage': masked_tokens / total_tokens if total_tokens > 0 else 0.0,
    #     'loss_mask/batch_entries_with_mask': (loss_mask.sum(-1) > 0).sum().item()
    # })

    # return batch, metrics

def _is_subsequence(sequence, subsequence, start_pos):
    """
    Helper method to check if a subsequence exists at a given position in a sequence.
    
    Parameters:
        sequence: The main sequence to check within
        subsequence: The subsequence to look for
        start_pos: The position to start checking from
        
    Returns:
        Boolean indicating whether the subsequence exists at start_pos
    """
    if start_pos + len(subsequence) > len(sequence):
        return False
    
    for i, token_id in enumerate(subsequence):
        if sequence[start_pos + i] != token_id:
            return False
    
    return True

if __name__ == "__main__":
    import transformers
    import torch

    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # Example tensor to test the loss mask creation
    example_response = torch.load("/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/verl/_debug_output/debug_responses_tensor.pt")
    _create_loss_mask(
        tokenizer,
        example_response,
        start_marker="<|im_start|>assistant",
        end_marker="<|im_end|>")