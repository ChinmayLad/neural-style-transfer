import torch.nn as nn
from torchvision.models import vgg16

class VGG(nn.Module):
    """
    The VGG class imports a pretrained VGG-16 model from torchvision library.
    This class is used for calculating the feature reconstruction loss (Content Loss)
    and style reconstruction loss (Style Loss).
    style_layers array consist of layers whose output are used for style representation.
    structure_layers array consist of layer whose output are used for content-representation
    and is one of the higher layer as discussed in the paper.
    """
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg16 = vgg16(pretrained=True).eval()
        self.style_layers = ['3', '8', '15', '22']
        self.structure_layers = ['15']
    
    def forward(self, x):
        return vgg16(x)
    
    def _get_layer_outputs(self, x, layers):
        out = []
        for i, l in self.vgg16.features._modules.items():
            x = l(x)
            if i in layers:
                out.append(x)
            if i == layers[-1]:
                break
            
        return out

    def style(self, x):
        return self._get_layer_outputs(x, self.style_layers)

    def structure(self, x):
        return self._get_layer_outputs(x, self.structure_layers)[0]

    def feature_reconstruction_loss(self, pred, target, weights):
        if target.shape[0] != pred.shape[0]:
            target = torch.cat([target for _ in range(pred.shape[0])], 0)
        phi_pred = self.structure(pred)
        phi_target = self.structure(target)
        squared_error = (phi_pred - phi_target)**2
        return weights*torch.mean(squared_error, dim=(1,2,3))

    def style_reconstruction_loss(self, pred, target, weights):
        if target.shape[0] != pred.shape[0]:
            target = torch.cat([target for _ in range(pred.shape[0])], 0)
        phi_pred = self.style(pred)
        phi_target = self.style(target)
        s = torch.empty(pred.shape[0]).fill_(0.0).requires_grad_(True)
        if torch.cuda.is_available():
            s = s.cuda()
        for w, p, t in zip(weights, phi_pred, phi_target):
            gm_pred = gram_matrix(p)
            gm_target = gram_matrix(t)
            squared_error =  w * torch.sum((gm_pred - gm_target)**2, dim=(1, 2))
            s = torch.add(s, squared_error)
        return s

if __name__ == "__main__":
    vgg = VGG()
    print(vgg)