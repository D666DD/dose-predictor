import copy
import os
import warnings
import scipy.io as sio
from absl import app, flags
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from diffusion_struc_skip import GaussianDiffusionTrainer, GaussianDiffusionSampler
# from model import MambaUnet
from Dataset.dataset import Train_Data, Valid_Data
from Dataset.datasets_new2_dm import MyDataset
from vim.mamba_struc_unet_2u import MambaUnet
from vim.mamba_struc_unet_2u import two_unet


FLAGS = flags.FLAGS
flags.DEFINE_bool("train", False, help="train from scratch")
flags.DEFINE_bool("continue_train", True, help="train from scratch")

# UNet
flags.DEFINE_integer("ch", 64, help="base channel of UNet")
flags.DEFINE_multi_integer("ch_mult", [1, 2, 2, 4, 4], help="channel multiplier")
flags.DEFINE_multi_integer("attn", [1], help="add attention to these levels")
flags.DEFINE_integer("num_res_blocks", 2, help="# resblock in each level")
flags.DEFINE_float("dropout", 0.0, help="dropout rate of resblock")

# Gaussian Diffusion
flags.DEFINE_float("beta_1", 1e-4, help="start beta value")
flags.DEFINE_float("beta_T", 0.02, help="end beta value")
flags.DEFINE_integer("T", 1000, help="total diffusion steps")
flags.DEFINE_enum(
    "mean_type", "epsilon", ["xprev", "xstart", "epsilon"], help="predict variable"
)
flags.DEFINE_enum(
    "var_type", "fixedlarge", ["fixedlarge", "fixedsmall"], help="variance type"
)

# Training
flags.DEFINE_float("lr", 1e-4, help="target learning rate") #1e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("img_size", 128, help="image size")
flags.DEFINE_integer("batch_size", 16, help="batch size")
flags.DEFINE_integer("num_workers", 2, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")

# Logging & Sampling
flags.DEFINE_string("DIREC", "OpenKBP_NEW2_dm_2u_aug_1500_best_net_8", help="name of your project")
flags.DEFINE_integer("sample_size", 128, "sampling size of images")
flags.DEFINE_integer(
    "max_epoch",
    1500,
    help="frequency of saving checkpoints, 0 to disable during training",
)

device = torch.device("cuda:0")


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def train():
    # dataset
    tr_train = MyDataset("train")
    trainloader = DataLoader(
        tr_train,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    va_train = MyDataset("val")
    validloader = DataLoader(
        va_train,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    # model setups
    
    net_model = two_unet(
        T_=FLAGS.T,
        ch_=FLAGS.ch,
        ch_mult_=FLAGS.ch_mult,
        attn_=FLAGS.attn,
        num_res_blocks_=FLAGS.num_res_blocks,
        dropout_=FLAGS.dropout,
        dim1=16,
        dim2=32,
        inp_channels1 = 1,
        inp_channels2 = 2,
        inp_struc_channels1=21,
        inp_struc_channels2=21,
        out_channels1=1,
        out_channels2=1,
    )
    ema_model = copy.deepcopy(net_model)

    init_lr = FLAGS.lr
    lr = FLAGS.lr
    best_loss = 1000.0
    optim = torch.optim.Adam(net_model.parameters(), lr=lr)
    '''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    '''
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T
    ).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model,
        FLAGS.beta_1,
        FLAGS.beta_T,
        FLAGS.T,
        FLAGS.img_size,
        FLAGS.mean_type,
        FLAGS.var_type,
    ).to(device)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size))# / 1024 / 1024))

    if FLAGS.continue_train:
        checkpoint = torch.load(
            "./Save/"
            + FLAGS.DIREC
            + "/hxdose-struc-mamba-unet"
            + "/model_latest.pkl",
            #+ "/model_epoch_800.pkl",
            #+ "/model_epoch_1000.pkl",
            map_location="cuda:0",
        )
        net_model.load_state_dict(checkpoint["net_model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optim.load_state_dict(checkpoint["optim"])
        restore_epoch = checkpoint["epoch"]
        best_loss = checkpoint["loss"]
        #scheduler.load_state_dict(checkpoint["scheduler"])
        print("Finish loading model")
    else:
        restore_epoch = 0

    if not os.path.exists("Loss"):
        os.makedirs("Loss")

    tr_ls = []
    if FLAGS.continue_train:
        readmat = sio.loadmat("./Loss/" + FLAGS.DIREC)
        load_tr_ls = readmat["loss"]
        #for i in range(restore_epoch):
            #tr_ls.append(load_tr_ls[0][i])
        print("Finish loading loss!")

    for epoch in range(restore_epoch, FLAGS.max_epoch):
        with tqdm(trainloader, unit="batch") as tepoch:
            tmp_tr_loss = 0
            tmp_tr_loss1 = 0
            tmp_tr_loss2 = 0
            tr_sample = 0
            net_model.train()
            for data, target, mask, cbct_path, ct_path in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                optim.zero_grad()
                condition = data.to(device)
                x_0 = target.to(device)
                loss, loss1, loss2 = trainer(x_0, condition)
                tmp_tr_loss += loss.item()
                tmp_tr_loss1 += loss1.item()
                tmp_tr_loss2 += loss2.item()
                tr_sample += 1
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
                optim.step()
                ema(net_model, ema_model, FLAGS.ema_decay)

                #tepoch.set_postfix({"Loss": loss.item()})
                tepoch.set_postfix({"Loss": tmp_tr_loss / tr_sample,
                                    "Loss1": tmp_tr_loss1 / tr_sample,
                                    "Loss2": tmp_tr_loss2 / tr_sample,
                })
                
        epoch_train_loss = tmp_tr_loss / tr_sample
        tr_ls.append(epoch_train_loss)
        #scheduler.step(epoch_train_loss)
        
        sio.savemat("./Loss/" + FLAGS.DIREC + ".mat", {"loss": tr_ls})

        if not os.path.exists("Train_Output/" + FLAGS.DIREC):
            os.makedirs("Train_Output/" + FLAGS.DIREC)
            
        if not os.path.exists("Save/" + FLAGS.DIREC):
            os.makedirs("Save/" + FLAGS.DIREC)
            
        if not os.path.exists("./Save/" + FLAGS.DIREC + "/hxdose-struc-mamba-unet"):
            os.makedirs("./Save/" + FLAGS.DIREC + "/hxdose-struc-mamba-unet")
        val_loss = 0.0
        val_samples = 0
        net_model.eval()
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                for data, target, mask, cbct_path, ct_path in validloader:
                    condition = data.to(device)
                    x_0 = target.to(device)
                    loss, _, _ = trainer(x_0, condition)
                    val_loss += loss.item()
                    val_samples += 1
                final_loss = val_loss / val_samples
                print('final_loss:', final_loss)
                if final_loss < best_loss:
                    best_loss = final_loss
                    ckpt = {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "optim": optim.state_dict(),
                        "epoch": epoch + 1,
                        "loss": final_loss,
                        #"scheduler": scheduler.state_dict()
                        # 'x_T': x_T,
                    }
                    torch.save(
                        ckpt,
                        "./Save/"
                        + FLAGS.DIREC
                        + "/hxdose-struc-mamba-unet"
                        + "/model_best.pkl",
                    )
                    

        # save
        if not os.path.exists("Save/" + FLAGS.DIREC):
            os.makedirs("Save/" + FLAGS.DIREC)
        ckpt = {
            "net_model": net_model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch + 1,
            "loss": best_loss,
            #"scheduler": scheduler.state_dict()
            # 'x_T': x_T,
        }
        if not os.path.exists("./Save/" + FLAGS.DIREC + "/hxdose-struc-mamba-unet"):
            os.makedirs("./Save/" + FLAGS.DIREC + "/hxdose-struc-mamba-unet")
        if (epoch + 1) % 100 == 0:
            torch.save(
                ckpt,
                "./Save/"
                + FLAGS.DIREC
                + "/hxdose-struc-mamba-unet"
                + "/model_epoch_"
                + str(epoch + 1)
                + ".pkl",
            )
        torch.save(
            ckpt,
            "./Save/"
            + FLAGS.DIREC
            + "/hxdose-struc-mamba-unet"
            + "/model_latest.pkl",
        )


def test():
    # dataset
    va_train = MyDataset("test")
    validloader = DataLoader(
        va_train,
        batch_size=FLAGS.sample_size,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    # model setup
    net_model = two_unet(
        T_=FLAGS.T,
        ch_=FLAGS.ch,
        ch_mult_=FLAGS.ch_mult,
        attn_=FLAGS.attn,
        num_res_blocks_=FLAGS.num_res_blocks,
        dropout_=FLAGS.dropout,
        dim1=16,
        dim2=32,
        inp_channels1 = 1,
        inp_channels2 = 2,
        inp_struc_channels1=21,
        inp_struc_channels2=21,
        out_channels1=1,
        out_channels2=1,
    )
    ema_model = copy.deepcopy(net_model)

    ema_sampler = GaussianDiffusionSampler(
        #ema_model,
        net_model,
        FLAGS.beta_1,
        FLAGS.beta_T,
        FLAGS.T,
        FLAGS.img_size,
        FLAGS.mean_type,
        FLAGS.var_type,
    ).to(device)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    checkpoint = torch.load(
        "./Save/"
        + "OpenKBP_NEW2_dm_2u_aug"
        #+ FLAGS.DIREC
        + "/"
        + "hxdose-struc-mamba-unet/"
        + "/model_best.pkl"
        #+ "/model_latest.pkl"
        #+ "/model_epoch_600.pkl"
    )
    net_model.load_state_dict(checkpoint["net_model"])
    ema_model.load_state_dict(checkpoint["ema_model"])
    restore_epoch = checkpoint["epoch"]
    print("Finish loading model")

    # output = np.zeros((9118, 6, 128, 128))  # example size, please change based on your data
    # lr = np.zeros((9118, 256, 256))
    # hr = np.zeros((9118, 256, 256))
    if not os.path.exists("Test_Output/" + FLAGS.DIREC):
        os.makedirs("Test_Output/" + FLAGS.DIREC)
    if not os.path.exists(
                    "./Test_Output/"
                    + FLAGS.DIREC
                    + "/hxdose-struc-mamba-unet-all-image"
                ):
        os.makedirs(
                        "./Test_Output/"
                        + FLAGS.DIREC
                        + "/hxdose-struc-mamba-unet-all-image"
                    )
    if not os.path.exists(
                    "./Test_Output/" + FLAGS.DIREC + "/hxdose-struc-mamba-unet-all"
                ):
        os.makedirs(
                        "./Test_Output/"
                        + FLAGS.DIREC
                        + "/hxdose-struc-mamba-unet-all"
                    )
    net_model.eval()
    count = 0
    with torch.no_grad():
        # with tqdm(validloader, unit="batch") as tepoch:
        for data, target,mask,cbct_path,ct_path in tqdm(validloader):
            count += 1
            if count == 1000:
                break
            skip = 0
            folder_path =  "./Test_Output/"+ FLAGS.DIREC+ "/"+ "hxdose-struc-mamba-unet-all/"
            print(folder_path)
            for file_name in os.listdir(folder_path):
                target_name = "pt_" + str(ct_path[0]).split("_")[1] + "_dose_" + str(ct_path[0]).split("_")[2]
                if target_name in file_name:
                    skip = 1
                    break

            if skip == 1:
                continue
            condition = data.to(device)
            length = data.shape[0]
            x_T = torch.randn(length, 1, FLAGS.img_size, FLAGS.img_size)
            #min_val = x_T.min()
            #max_val = x_T.max()
            #x_T = (x_T - min_val) / (max_val - min_val)
            x_T = x_T.to(device)
            import time

            # start=time.time()
            print(FLAGS.DIREC)
            x_0 = ema_sampler(x_T, condition)
            
            print(len(x_0))
            print(len(x_0[4]))
            for i in range(FLAGS.sample_size):
              print("./Test_Output/"
                + FLAGS.DIREC
                + "/"
                + "hxdose-struc-mamba-unet-all/"
                + "pt_"
                + str(ct_path[i]).split("_")[1]
                + "_dose_"
                + str(ct_path[i]).split("_")[2]
                )
              #x_save = x_0[4][i].detach().cpu().numpy()
              x_save = x_0[i][0].detach().cpu().numpy()
              np.save(
                "./Test_Output/"
                + FLAGS.DIREC
                + "/"
                + "hxdose-struc-mamba-unet-all/"
                + "pt_"
                + str(ct_path[i]).split("_")[1]
                + "_dose_"
                + str(ct_path[i]).split("_")[2]
                ,x_save)


def train_main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action="ignore", category=FutureWarning)
    if FLAGS.train:
        train()
    test_main(argv)


def test_main(argv):
    test()


if __name__ == "__main__":
    app.run(train_main)
