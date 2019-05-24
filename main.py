import argparse 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import datetime
import time
from critic import Critic
from generator import Generator
from torch.autograd import grad
from torch.utils import data
import torch.nn.functional as f
import torch
import os
from tqdm import tqdm
from torch.distributions.kl import kl_divergence
import torchvision.models as models

parser = argparse.ArgumentParser(description='Confidence Calibrated Classifiers for Detecting Out-of-Distribution Samples')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in dataset')
parser.add_argument('--beta', type=int, default=1, help='Penalty weight for confidence loss')
parser.add_argument('--save_model', type=int, default=0, help='Save checkpoints of model while training')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--output_dir', type=str, default='~/', help='Directory to save models and logs')
parser.add_argument('--use_confidence_loss', type=int, default=1, help='Use confidence loss. If 0, then only CrossEntropy is used.')
parser.add_argument('--gradient_updates', type=int, default=3500, help='Number of gradient updates.')
opt = parser.parse_args()

torch.cuda.set_device(0)

m = opt.batch_size
latent_size = 100

transform = transforms.Compose([transforms.ToTensor()])
cifar10_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform = transform)
cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform = transform)
svhn_testset = datasets.SVHN(root='./data', split='test', download=True, transform = transform)

trainloader = data.DataLoader(cifar10_trainset, shuffle=True, batch_size = m, drop_last = True)
testloader = data.DataLoader(cifar10_testset, shuffle=True, batch_size = m, drop_last = True)
svhntestloader = data.DataLoader(svhn_testset, shuffle=True, batch_size = m, drop_last = True)

latent_distr = torch.distributions.normal.Normal(0, 1)

# Networks      
crit = Critic()
gen = Generator(latent_size)
classifier = models.vgg13(pretrained=False)
# adjust final layer to handle 10 classes
classifier.classifier._modules['6'] = torch.nn.Linear(4096, 10)
classifier.train()
crit.cuda()
gen.cuda()
classifier.cuda()

adversarial_loss = torch.nn.BCELoss()
neg_logl = torch.nn.NLLLoss()

optimizer = torch.optim.Adam(crit.parameters(), lr = 0.0001, betas=(0.5, 0.999))
optimizer_gen = torch.optim.Adam(gen.parameters(), lr = 0.0001, betas=(0.5, 0.999))
optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr = 0.0001, betas=(0.5, 0.999))

scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_gen, gamma=0.999)
scheduler_c = torch.optim.lr_scheduler.ExponentialLR(optimizer_classifier, gamma=0.999)

t = 0

S_max = opt.gradient_updates
beta = opt.beta

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
output_dir = opt.output_dir + st + "ood/"
os.mkdir(output_dir)
writer = SummaryWriter(output_dir)

def save_model(crit, gen, classifier, optimizer, optimizer_gen, optimizer_classifier, t):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S') #  String with current date

    torch.save({
                'Critic_state_dict': crit.state_dict(),
                'Gen_state_dict': gen.state_dict(),
                'Classifier_state_dict': classifier.state_dict(),
                'optimizerCritic_state_dict': optimizer.state_dict(),
                'optimizerGen_state_dict': optimizer_gen.state_dict(),
                'optimizerClass_state_dict': optimizer_classifier.state_dict(),
                't': (t) 
                }, output_dir + st)

epoch = 0
train_iter = iter(trainloader)
test_iter = iter(testloader)
for t in tqdm(range(0, S_max)):

    if opt.use_confidence_loss:
        # if not, the GAN is not needed
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(trainloader)
            data = next(train_iter)
            epoch += 1

        data_tmp = data[0]

        # labels
        real = torch.Tensor(data_tmp.size(0), 1).fill_(1.0).cuda()
        generated = torch.Tensor(data_tmp.size(0), 1).fill_(0.0).cuda()

        # Discriminator
        optimizer.zero_grad()
        x_real = data_tmp
        x_real = x_real.cuda()

        zeros = torch.zeros(m,1).cuda() 
        latent_sample = latent_distr.sample(torch.Size([latent_size, m])).view(m, latent_size)
        loss = 0

        # generate samples
        z_lat = latent_sample.view(-1, latent_size).cuda() 

        gen_data = gen(z_lat)

        noise1 = torch.randn(x_real.size()).cuda()
        critic_img = x_real

        eval_real = crit(critic_img)
        x_gen = gen_data.detach()

        eval_gen = crit(x_gen) 

        real_loss = adversarial_loss(eval_real, real)

        fake_loss = adversarial_loss(eval_gen, generated)

        loss = real_loss + fake_loss
        loss = loss.mean()
        loss.backward()

        optimizer.step()
            
        writer.add_scalar("loss/crit", loss.detach(), t)

        # Generator
        optimizer_gen.zero_grad()

        zeros = torch.zeros(m,1).cuda() 
        #x_real = norm_distr.sample(torch.Size([m])).view(m,1)
        latent_sample = latent_distr.sample(torch.Size([latent_size, m])).view(m, latent_size)
        # just 1/10 per class for cifar10
        uniform_dist = torch.Tensor(data_tmp.size(0), opt.num_classes).fill_((1./opt.num_classes)).requires_grad_().cuda()
        loss = 0

        # generate samples
        z_lat = latent_sample.view(-1, latent_size).cuda() 
        
        gen_data = gen(z_lat)

        eval_gen = crit(gen_data)
        
        fake_loss = adversarial_loss(eval_gen, real)

        # confidence term
        out_of_distr_predict = f.log_softmax(classifier(gen_data).detach(), dim=1)
        conf_loss = f.kl_div(out_of_distr_predict, uniform_dist)

        loss = fake_loss + beta*conf_loss
        loss = loss.mean()
        loss.backward()

        optimizer_gen.step()
        writer.add_scalar("loss/gen",loss.detach(), t)

    # Classifier
    optimizer_classifier.zero_grad()

    try:
        data = next(train_iter)
    except StopIteration:
        train_iter = iter(trainloader)
        data = next(train_iter)
        epoch += 1

    data_tmp = data[0].cuda()
    labels = data[1].cuda()

    conf_loss = 0
    if opt.use_confidence_loss:
        uniform_dist = torch.Tensor(data_tmp.size(0), opt.num_classes).fill_((1./opt.num_classes)).requires_grad_().cuda()

        latent_sample = latent_distr.sample(torch.Size([latent_size, m])).view(m, latent_size)
        z_lat = latent_sample.view(-1, latent_size).cuda()
        gen_data = gen(z_lat)

        out_of_distr_predict = f.log_softmax(classifier(gen_data), dim=1)
        conf_loss = f.kl_div(out_of_distr_predict, uniform_dist)*opt.num_classes
        writer.add_scalar("loss/conf_loss",conf_loss, t)

    in_distr_predict = f.log_softmax(classifier(data_tmp), dim=1)
     
    cross_entropy = neg_logl(in_distr_predict, labels)

    writer.add_scalar("loss/cross_entropy",cross_entropy, t)

    loss = cross_entropy + beta*conf_loss
    loss = loss.mean()
    loss.backward()

    optimizer_classifier.step()
    writer.add_scalar("loss/classifier",loss.detach(), t)

    # Calculate test loss
    with torch.no_grad():
        try:
            data = next(test_iter)
        except StopIteration:
            test_iter = iter(testloader)
            data = next(test_iter)

        data_tmp = data[0].cuda()
        labels = data[1].cuda()
        conf_loss = 0

        if opt.use_confidence_loss:
            uniform_dist = torch.Tensor(data_tmp.size(0), opt.num_classes).fill_((1./opt.num_classes)).requires_grad_().cuda()
            latent_sample = latent_distr.sample(torch.Size([latent_size, m])).view(m, latent_size)
            z_lat = latent_sample.view(-1, latent_size).cuda()
            gen_data = gen(z_lat)   
            conf_loss = f.kl_div(out_of_distr_predict, uniform_dist)*opt.num_classes
            out_of_distr_predict = f.log_softmax(classifier(gen_data), dim=1)
            writer.add_scalar("loss/test_conf_loss",conf_loss, t)

        in_distr_predict = f.log_softmax(classifier(data_tmp), dim=1)
        cross_entropy = neg_logl(in_distr_predict, labels)
        writer.add_scalar("loss/test_cross_entropy",cross_entropy, t)
        loss = cross_entropy + beta*conf_loss
        loss = loss.mean()
        writer.add_scalar("loss/test_classifier",loss.detach(), t)

    if t % 200 == 0 and opt.use_confidence_loss:
        print(t)
        gen_x = gen_data[0,:,:,:].cpu() 
        name = str(t)
        writer.add_image(name, gen_x, t)

    if epoch % 10 == 0:
        if opt.use_confidence_loss:
            scheduler_d.step()
            scheduler_g.step()    
            larg_lr_crit = [x["lr"] for x in list(optimizer.param_groups)]
            larg_lr_crit = max(larg_lr_crit)
            writer.add_scalar("loss/crit_lr", larg_lr_crit)

        scheduler_c.step()
        larg_lr_clas = [x["lr"] for x in list(optimizer_classifier.param_groups)]
        larg_lr_clas = max(larg_lr_clas)
        writer.add_scalar("loss/clas_lr", larg_lr_clas)
        if opt.save_model == 1 and opt.use_confidence_loss:
            save_model(crit, gen, classifier, optimizer, optimizer_gen, optimizer_classifier, t)

with torch.no_grad():
    # Final Test accuracy
    classifier.eval()
    total = 0
    correct = torch.tensor(0.)
    for t, data in enumerate(testloader):
        total += m
        data_tmp = torch.tensor(data[0]).cuda()
        labels = torch.tensor(data[1]).cuda()
        in_distr_predict = f.log_softmax(classifier(data_tmp), dim=1)
        predicted_idx = in_distr_predict.max(1)[1]  
        correct += predicted_idx.eq(labels).cpu().sum()

    acc = 100.*correct/total
    print("\nCorrect: ", correct, " Total: ", total, " Accuracy: ", acc.item(), "%\n")

    # Detection Accuracy - "This metric corresponds to the maximum classification probability over all possible thresholds"
    # Simplified version - small number of thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    maximum_classification_probability = 0
    # px_is_from_Pin = px_is_from_Pout = 0.5
    # 155 test data batches in cifar10
    # 405 in svhn
    zeros = torch.zeros(1).cuda()
    ones = torch.ones(1).cuda()
    minimum = 999999999
    for threshold in thresholds:
        total = 0
        in_distr_tot = 0
        out_distr_tot = 0
        threshold = torch.tensor(threshold).cuda()
        for in_data, out_data in zip(testloader, svhntestloader):
            total += m
            # in distribution
            data_tmp = torch.tensor(in_data[0]).cuda()
            in_distr_predict = f.softmax(classifier(data_tmp), dim=1)
            max_probability = in_distr_predict.max(1)[0]
            in_distr_below_threshold = torch.where(max_probability <= threshold, max_probability, zeros)
            # for counting
            in_distr_below_threshold = torch.where(in_distr_below_threshold == 0, in_distr_below_threshold, ones) 
            in_distr_tot += in_distr_below_threshold.sum()
            
            #out of distribution
            data_tmp = torch.tensor(out_data[0]).cuda()
            out_distr_predict = f.softmax(classifier(data_tmp), dim=1)
            max_probability = out_distr_predict.max(1)[0]
            out_distr_above_threshold = torch.where(max_probability > threshold, max_probability, zeros)
            # for counting
            out_distr_above_threshold = torch.where(out_distr_above_threshold == 0, out_distr_above_threshold, ones)
            out_distr_tot += out_distr_above_threshold.sum()

        print("In Distribution total: ", in_distr_tot, " Out of Distribution total: ", out_distr_tot, " Total: ", total)
        summed_probabilities = 0.5*(in_distr_tot + out_distr_tot)/total
        if summed_probabilities < minimum:
            minimum = summed_probabilities
            classification_probability = 1 - summed_probabilities
            if classification_probability > maximum_classification_probability:
                maximum_classification_probability = classification_probability

    print("Maximum Classification Probability: ", maximum_classification_probability)

writer.close()

