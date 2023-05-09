import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision.transforms.functional import gaussian_blur
from torchvision.utils import save_image

from tqdm import trange, tqdm
from pathlib import Path
from pickle import load
import matplotlib.pyplot as plt
from imagenet import dic
from einops import rearrange


class adversarial():
    def __init__(self, device, loader, model, criterion, epsilons, save_samples, img_path=None) -> None:
        
        self.epsilons = [0, 16., 32., 48., 64., 80.] if epsilons is None else epsilons 
        self.epsilons = [i / 255. for i in self.epsilons]
        print(f'normalized epsilons: {self.epsilons}')
        self.loader = loader
        self.model = model.to(device)
        self.model.eval()

        self.device = device
        self.criterion = criterion
        self.save = save_samples
        self.per_eps = []
        for _ in range(len(self.epsilons)):
            self.per_eps.append([])
        if img_path:
            self.img_path = Path(img_path)
            self.img_path.mkdir(exist_ok=True)


        self.target_index = self.select_index()

        # Pytorch pretrained normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)
        self.norm = Normalize(mean=self.mean, std=self.std)
        self.mean = rearrange(self.mean, '(a b c d) -> a b c d', a=1, b=3, c=1, d=1)
        self.std = rearrange(self.std, '(a b c d) -> a b c d', a=1, b=3, c=1, d=1)
        
    
    def find_adversarial(self):
        pass

    
    def attack(self, epsilon, image, original_img, signed_gradient, eps_neigbor_clip=False):
        # note that image == original_img in the first iteration

        # denormalize, then the range of pixel will be in [0, 1]
        image = image * self.std + self.mean
        original_img = original_img * self.std + self.mean

        if not eps_neigbor_clip:
            return self.norm(torch.clamp(original_img + epsilon * signed_gradient, min=0, max=1))
        else:
            perturbed = image + (signed_gradient / 255.)
            return self.norm(self.eps_neighbor_clipping(perturbed, original_img, epsilon))
    

    def eps_neighbor_clipping(self, perturbed_img: torch.Tensor, original_img: torch.Tensor, epsilon) -> torch.Tensor:
        return torch.min(torch.tensor(1), torch.min(original_img + epsilon, torch.max(torch.tensor(0), torch.max(original_img - epsilon, perturbed_img))))
        # torch.tensor(1), torch.tensor(0) will be broadcasted

    def select_index(self):

        lbl_dict = dict(
            n01440764='tench',
            n02102040='English_springer',
            n02979186='cassette_player',
            n03000684='chain_saw',
            n03028079='church',
            n03394916='French_horn',
            n03417042='garbage_truck',
            n03425413='gas_pump',
            n03445777='golf_ball',
            n03888257='parachute'
        )
        
        with open('image_to_imagenette.pickle', 'rb') as f:
            self.mapping = load(f)
        target_idx = []
        for key in lbl_dict.keys():
            target_idx += int(self.mapping[key][0]), # e.g. mapping['n03888257'] = ('701', 'parachute')
            assert lbl_dict[key] == self.mapping[key][1], f'mapped {lbl_dict[key]} to {self.mapping[key][1]}, key = {key}'

        return target_idx



class Attacker(adversarial):
    def __init__(self, device, loader, model, criterion, epsilons, save_samples, img_path=None, iterative=False, least=False, process_perturbed=False) -> None:
        super().__init__(device, loader, model, criterion, epsilons, save_samples, img_path)

        self.iterative = iterative
        self.least = least
        self.count_correct = True
        self.process_perturbed = process_perturbed

    def find_adversarial(self):
       
        accs = []        
        top5_accs = []
        with trange(len(self.epsilons)) as epsilons:
            
            for eps in epsilons:
                epsilons.set_postfix_str(f'find adversarial using epsilon = {self.epsilons[eps]:.5f}')
                correct = 0.0
                total = 0.0
                original_acc = 0.0
                top5_acc = 0.0
                original_top5_acc = 0.0
                repeat = 1
                if self.iterative:
                    repeat = min(int(255*self.epsilons[eps]+4), int(255*self.epsilons[eps]*1.25)) # from the paper

                for img, label in tqdm(self.loader, desc='iterating all images'):
                    total += 1
                    original_img = img.clone().to(self.device)
                    img, label = img.to(self.device), label.to(self.device)
                    img.requires_grad = True

                    output = self.model(img)

                    label = torch.tensor(self.target_index[label], device=self.device).unsqueeze(0)
                    
                    # if the model wrongly classified the sample, no need to attack it
                    prediction = output.argmax(dim=1)
                    if not label in torch.topk(output, 5)[1]:
                        # print('***', output.argmax(dim=1))
                        continue
                    if prediction == label:
                        original_acc += 1
                    original_top5_acc += 1

                    if self.least:
                        # find the least likely class
                        scores = output[:, self.target_index].detach()
                        least_label = scores.argmin()
                        least_label = torch.tensor(self.target_index[least_label], device=self.device).unsqueeze(0)

                    loss = 0
                    original_img = img.detach()
                    for i in range(repeat):
                        # if self.epsilons[eps] != 0: print('---', label, '---')
                        loss = self.criterion(output, least_label) if self.least else self.criterion(output, label)
                        loss.backward()
                        img = self.attack(self.epsilons[eps], img.detach(), original_img, torch.sign(img.grad.detach()), self.least, self.iterative)
                        img.requires_grad = True
                        if i == repeat - 1 and self.process_perturbed:
                            img = gaussian_blur(img, kernel_size=3, sigma=0.1)
                        output = self.model(img)

                    prediction = output.argmax(dim=1)
                    if prediction == label:
                        correct += 1
                        top5_acc += 1
                    elif label in torch.topk(output, 5)[1]:   
                        top5_acc += 1
                    else:
                        # print('---', output.argmax(dim=1))
                        
                        img = img * self.std + self.mean
                        original_img = original_img * self.std + self.mean
                        if isinstance(self.save, int):
                            if len(self.per_eps[eps]) < self.save:
                                self.per_eps[eps].append((original_img, img, f'label: {dic[str(label.int().item())]}, prediction: {dic[str(output.argmax(dim=1).int().item())]}'))
                        elif self.save:
                            if len(self.per_eps[eps]) < 2: # save 2 imgs for all eps if self.save == true
                                self.per_eps[eps].append((original_img, img, f'label: {dic[str(label.int().item())]}, prediction: {dic[str(output.argmax(dim=1).int().item())]}'))

                accs.append(correct / total)
                top5_accs.append(top5_acc / total)
                print(f'\n for eps = {self.epsilons[eps]:.5f}, acc before attack is {original_acc / total:.5f}')
                print(f'for eps = {self.epsilons[eps]:.5f}, acc after attack is {correct / total:.5f}')
                print(f'for eps = {self.epsilons[eps]:.5f}, top5 acc before attack is {original_top5_acc / total:.5f}')
                print(f'for eps = {self.epsilons[eps]:.5f}, top5 acc after attack is {top5_acc / total:.5f}')
                
                folder = 'FGSM'
                if self.least and self.iterative:
                    folder = 'least'
                elif self.iterative:
                    folder = 'iter'

                for i in self.per_eps:
                    path = self.img_path / f'{folder}_{self.model.name}_{int(self.epsilons[eps] * 255)}'
                    path.mkdir(exist_ok=True, parents=True)
                    for imgs in range(len(i)):
                        save_image(i[imgs][0], path / f'{imgs}_original.jpg')
                        save_image(i[imgs][1], path / f'{imgs}_adv.jpg')
                        with open(path / f'{imgs}.txt', 'w') as f:
                            f.write(i[imgs][2]) # 'index': 'dir name', 'class name'

        return accs, top5_accs

    def attack(self, epsilon, image, original, signed_gradient, least, not_fgsm):
    
        if least:
            signed_gradient *= -1

        return super().attack(epsilon, image, original, signed_gradient, not_fgsm)
        