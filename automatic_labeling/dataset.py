#!/usr/bin/python3
RESULT = "result"

def _truncate(text, limit):
    if not limit:
        return text

    start, end = limit
    if len(text) <= start + end:
        return text

    return text[:start] + "[...]" + text[-end:]

def build_context_window(
    turns,
    end_idx,
    prompt_turns = 5, 
    prompt_limit = (250, 250),
    response_turns = 1, 
    response_limit = None,
    information_to_add = None,
    information_from = 'llm'
):
    '''
    Building chat context window.
    
    :param turns:               The complete chat history
    :param end_idx:             The end index for the window, not included
    :param prompt_turns:        Number of prompt turns to be used (else: "...")
    :param prompt_limit:        Truncate lengths for prompts
    :param response_turns:      Number of response turns to be used (else: "...")
    :param response_limit:      Truncate lengths for responses
    :param information_to_add:  Field names of where to take information from 
                                (turn['labels'][information_from][information_to_add])
    :param information_from:    From where to take the information (llm, human)
    '''
    if information_to_add is None:
        information_to_add = []
        
    window_length = max(prompt_turns, response_turns)
    window = turns[max(0, end_idx - window_length):end_idx]
    
    result = []
    
    prompt_indices = set(range(len(window) - prompt_turns, len(window)))
    response_indices = set(range(len(window) - response_turns, len(window)))
    
    for i, turn in enumerate(window):
        if i in prompt_indices:
            user_text = _truncate(turn["prompt"], prompt_limit)
        else:
            user_text = "..."
            
        if i in response_indices:
            assistant_text = _truncate(turn["response"], response_limit)
        else:
            assistant_text = "..."

        context = ", ".join([
            f"{info} = {turn['labels'][information_from].get(info, {}).get(RESULT, 'n/a')}"
            for info in information_to_add
        ])
        result.append(f"Turn {i}: " + context)
        result.append(f"User: {user_text}\nAssistant: {assistant_text}")
    return "\n".join(result)