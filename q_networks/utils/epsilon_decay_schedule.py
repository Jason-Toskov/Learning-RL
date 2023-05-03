def linear_decay(current_step: int, eps_start: float, eps_end: float, final_step: int):
    if current_step > final_step:
        return eps_end
    return current_step * (eps_end - eps_start) / final_step + eps_start
