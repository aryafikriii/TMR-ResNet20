# TMR Implementation on ResNet20

This project demonstrates the implementation of Triple Modular Redundancy (TMR) on the ResNet20 model using PyTorch framework. TMR is a fault-tolerant technique that improves the reliability of digital systems by triplicating the hardware and running computations in a redundant manner.

## Prerequisites

- Python 3.x
- PyTorch

## Installation

1. Clone the repository:

```shell
git clone https://github.com/aryafikriii/TMR-ResNet20
```

2. Install the required dependencies:

```shell
pip install -r requirements.txt
```

## Usage

1. Import the necessary libraries:

```python
import torch
import torchvision.models as models
```

2. Load the ResNet20 model:

```python
model = models.resnet20(pretrained=False)
```

3. Define the TMR wrapper class:

```python
class TMR_ResNet(nn.Module):
    def __init__(self, resnet):
        super(TMR_ResNet, self).__init__()
        self.model1 = resnet
        self.model2 = resnet
        self.model3 = resnet
        
    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        return out1, out2, out3
```

4. Wrap the ResNet20 model with the TMRWrapper:

```python
tmr_resnet = TMR_ResNet(resnet)
tmr_resnet = tmr_resnet.cuda()
```

5. Perform a forward pass on a sample input:

```python
input = torch.randn(1, 3, 32, 32)
out1, out2, out3 = tmr_resnet(input)
```

6. Combine the outputs using voting or any other fusion technique:

```python
final_output = (out1 + out2 + out3) / 3
```

7. Run the script and observe the results.

```shell
python tmr.py
```

## Contributing

Contributions are welcome! If you have any improvements or suggestions, please create a pull request or open an issue.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or need further assistance, please feel free to contact me at [aryafikriansyah@gmail.com].
