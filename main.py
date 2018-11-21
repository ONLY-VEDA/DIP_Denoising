import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import argparse
import time
from PIL import Image
from tensorboardX import SummaryWriter

from models import *
from utils import *
from dataset import SIDDDataset

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=30, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--is_gray', dest='is_gray', type=bool, default=False, help='gray image flag, if True, use gray image,default False')
parser.add_argument('--big_img', dest='big_img', type=bool, default=False, help='is image too big,so crop it')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint', dest='ckpt', default='', help='check point for restore')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--test_image', dest='test_image', default='test.jpg', help='image for inference')
parser.add_argument('--train_set', dest='train_set', default='./data/SIDD_Small/train', help='dataset for training')
parser.add_argument('--eval_set', dest='eval_set', default='./data/SIDD_Small/train', help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default='BSD68', help='dataset for testing')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    image_channel = 3
    if args.is_gray:
        image_channel = 1
    net = DnCNN(in_c=image_channel, out_c=image_channel)
    #net = IRCNN(in_c=image_channel, out_c=image_channel)
    #net = resnet50(channel=image_channel)
    if args.ckpt:
        net.load_state_dict(torch.load(args.ckpt))

    print(net)
    net.to(device)
    
    sidd_db = SIDDDataset(args.train_set)
    db_name = args.train_set.split('/')[-3]
    dataloader = DataLoader(sidd_db, batch_size = args.batch_size, 
            shuffle=True, num_workers = 8)

    #criterion = nn.MSELoss(reduction="sum")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9,0.999), eps=1e-08, weight_decay=0.0005)
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    gamma = 0.1
    decay_step = [5,10,25]

    start_time = time.time()
    writer = SummaryWriter()

    num_iter = 0
    for epoch in range(args.epoch):
        if epoch in decay_step:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * gamma 
        running_loss = 0.0 
        for i, data in enumerate(dataloader, 0):
            # get the inputs
            inputs, labels = data['noise'],data['clean']

            # pixels = inputs.shape[2]*inputs.shape[3]
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i+1) % 10 == 0:
                #print(optimizer.state_dict())
                print('[epoch: %d, iter: %5d] running loss: %.5f | time elapsed: %4.4f | learning rate: %.5f' % (epoch + 1, i + 1, running_loss/10,time.time() - start_time, float(optimizer.param_groups[0]['lr']) ))
                writer.add_scalar('loss', running_loss/10, num_iter)
                running_loss = 0.0

            num_iter += inputs.shape[0]
        
        save_name = args.ckpt_dir + "/" + db_name + '_' + net.__class__.__name__ + "_" + str(epoch) + ".ckpt"
        torch.save(net.state_dict(), save_name )
    writer.export_scalars_to_json('./logs.json')

def eval():
    image_channel = 3
    if args.is_gray:
        image_channel = 1
    net = DnCNN(in_c=image_channel, out_c=image_channel)
    print(net)

    net.load_state_dict(torch.load(args.ckpt, map_location='cuda:0'))
    print('model loaded %s'% args.ckpt)
    net.to(device)
    print('model to %s' % device)
    net.eval()

    sidd_db = SIDDDataset(args.eval_set)
    dataloader = DataLoader(sidd_db, batch_size = args.batch_size, 
            shuffle=False, num_workers = 8)

    cnt = 0
    sum_psnr = 0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data['noise'],data['clean']

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)
    
        #groundtruth = np.clip(255 * labels.numpy(), 0, 255).astype('uint8')
        #noisyimage = np.clip(255 * inputs.numpy(), 0, 255).astype('uint8')
        #outputimage = np.clip(255 * outputs.numpy(), 0, 255).astype('uint8')
        #psnr = cal_psnr(groundtruth, outputimage)
            
        groundtruth = torch.clamp(255 * labels, 0, 255)
        noisyimage = torch.clamp(255 * inputs, 0, 255)
        outputimage = torch.clamp(255 * outputs, 0, 255)

        psnr_pred = cal_psnr_torch(groundtruth, outputimage)
        print("batch %d predict PSNR: %.2f" % (i, psnr_pred))
        cnt += 1
        sum_psnr += psnr_pred.detach().cpu().numpy()
    print("average psnr: %.2f" % (sum_psnr/cnt ))

def inference():
    image_channel = 3
    if args.is_gray:
        image_channel = 1
    net = DnCNN(in_c=image_channel, out_c=image_channel)

    #net = IRCNN(in_c=3, out_c=3)
    #net = resnet50()
    print(net)

    net.load_state_dict(torch.load(args.ckpt, map_location='cuda:0'))
    print('model loaded %s'% args.ckpt)
    net.to(device)
    print('model to %s' % device)
    net.eval()

    image = Image.open(args.test_image)
    if not args.is_gray:
        image = image.convert('RGB')
        image = np.transpose(image,(2,0,1))
    else:
        image = image.convert('L')
        image = np.reshape(image, (1,image.size[0],image.size[1]))

    inputs =np.reshape(np.array(image, dtype="float32"),
            (1, image.shape[0], image.shape[1], image.shape[2]))  # extend one dimension
    
    # DEBUG
    if args.big_img:
        inputs = inputs[:,:,1000:1500,1000:1500]

    inputs = torch.from_numpy(inputs / 255.0).to(device)

    outputs = net(inputs)
    outputs = torch.clamp(255 * outputs, 0, 255)
    out = outputs.detach().cpu().numpy()[0]
    if not args.is_gray:
        out = np.transpose(out, (1, 2, 0))
    else:
        out = out[0]
    out = np.array(out, dtype="uint8")
    out = Image.fromarray(out)
    out.save('test_out.png')


        
def main():
    if args.phase == 'train':
        train()
    elif args.phase == 'eval':
        eval()
    elif args.phase == 'inference':
        inference()

if __name__ == "__main__":
    main()




