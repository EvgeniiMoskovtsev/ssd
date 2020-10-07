import torch
def load_model_weight():
    checkpoint = 'checkpoint_ssd300.pth.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model.eval().cpu()
    return model
def export_onnx_model(model, input_shape, onnx_path, input_names=None, output_names=None, dynamic_axes=None):
    inputs = torch.ones(*input_shape)
    model(inputs)
    torch.onnx.export(model, inputs, onnx_path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
if __name__ == "__main__":

    model = load_model_weight()
    for name, v in model.named_parameters():
        print(name)
    input_shape = (1, 3, 300, 300)
    onnx_path = "test.onnx"
    # input_names=['input']
    # output_names=['output']
    # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    export_onnx_model(model, input_shape, onnx_path)