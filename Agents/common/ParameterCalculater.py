def parametersCalculater(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params}")
    return total_params