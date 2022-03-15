import matplotlib.pyplot as plt
import torch
import math
import IPython
import MovingMNIST as MovingMNIST
from model_g import *
from circles_dataset import *

data = get_dataset(get_dataset_args())
train_set = torch.from_numpy(data['x']).transpose(0,1)
train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE, False)

# hierarchical net with three hidden layers
GEN_COORDS = 3
cause_sizes = [2, 0]
hidden_sizes = [0, 256]
net = GPC_net(b_size=BATCH_SIZE, dynamical_net=False, cause_sizes=cause_sizes, hidden_sizes=hidden_sizes)

# logging
errors, cov_h, errors_d1, errors_d2, errors_d3, cov_d1, cov_d2, cov_d3 = [[] for _ in range(8)]
err_g1, err_g2, err_g3, cov_g1, cov_g2, cov_g3 = [[] for _ in range(6)]
err_h1, err_h2, err_h3 = [], [], []

INTERPOLATE = False
INTERPOLATION_START = 25
INTERPOLATION_END = INTERPOLATION_START + GEN_COORDS-1
INTERPOLATION_EPOCH = 0
FINAL_EPOCH = 1

for epoch in range(FINAL_EPOCH):
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    for seq_id, seq in enumerate(train_loader):
        vid_in, vid_p1, vid_p2, vid_p3 = [], [], [], []
        vid_gc_cause, vid_gc_hidden = [], []
        vid_cause_coords, vid_gc_pred = [], []

        seq = torch.transpose(seq, 0,1)
        if seq_id == INTERPOLATION_EPOCH and epoch == FINAL_EPOCH-1:
            INTERPOLATE = True
        for id, data in enumerate(seq):
            data = data/2

            VIDEO = False
            if (id < INTERPOLATION_START): # optimize

                # initialize previous cause/hidden state = current cause/hidden state
                for layer in net.layers:
                    layer.dyn_model.state_diff(step=True) # move current to previous state


                if False:
                    # predict and decode state motion -> new prior on current cause/hidden state
                    net.layers[1].dyn_model.decode_dynamical_prediction() # predict hidden motion
                    net.layers[1].dyn_model.decode_dynamical_hidden(t=1) # decode hidden motion
                    net.layers[1].dyn_model.decode_dynamical_cause(t=1) # decode cause motion

                for update in range(UPDATES):
                    # track state motion
                    for layer in net.layers: layer.dyn_model.state_diff(step=False)

                    # step dynamical nets
                    _, _, _, _, _,  = net.layers[0].dyn_model.iterative_inference(data, updates=1)
                    eg1, eh1, ed1, cg1, cd1 = net.layers[1].dyn_model.iterative_inference(updates=1)
                    errors_d1, cov_d1 = errors_d1 + ed1, cov_d1 + cd1
                    err_g1, cov_g1 = err_g1 + eg1, cov_g1 + cg1
                    err_h1 = err_h1+eh1;

                    # step hierarchical net
                    e, _, _, ch, _ = net.iterative_inference(data, updates=1)
                    errors, cov_h = errors + e, cov_h + ch

                VIDEO = True
            elif INTERPOLATE:
                """Visualize generalized coordinates / encoded local trajectory"""
                # extrapolate encoded input in gen. coordinates
                time = id-INTERPOLATION_START+1
                cause_coords = net.layers[0].dyn_model.decode_dynamical_cause(t=time)
                vid_cause_coords.append(cause_coords)

                """Visualize dynamical prediction / motion of predicted local trajectory"""
                # predict and overwrite hidden state dynamics
                if False:
                    net.layers[1].dyn_model.decode_dynamical_prediction()

                # decode the predicted state motion # todo t
                net.layers[1].dyn_model.decode_dynamical_hidden(t=time)

                VIDEO = True

            # create video
            if VIDEO:
                input, preds = plot_2D(net=net, img_size=IMG_SIZE, plot=False)
                input = data.detach()[0]
                vid_in.append(input.detach().numpy().reshape([IMG_SIZE]))
                vid_p1.append(preds[0][0].reshape([IMG_SIZE]))
                #vid_p2.append(preds[1][0].reshape([IMG_SIZE]))
                #vid_p3.append(preds[2][0].reshape([IMG_SIZE]))

                state_coords = net.layers[0].dyn_model.layers[1].lower.states.cause_state[0].detach().numpy()
                hidd_coords = net.layers[0].dyn_model.layers[1].lower.states.hidd_state[0].detach().numpy()
                vid_gc_cause.append(state_coords)
                vid_gc_hidden.append(hidd_coords)

            if id == INTERPOLATION_END:
                print("seq_id", seq_id)
                break
        if seq_id == INTERPOLATION_EPOCH:
            break

            
            
            
            
""" Losses """
if True:
    plot_graph(errors, errors_d1, errors_d2, errors_d3, cov_d1, cov_d2, cov_d3,cov_g1,
               cov_g2, cov_g3, err_g1, err_g2, err_g3, err_h1, err_h2, err_h3, cov_h,
               hierarchical=True, g_coords=True, dynamical=False)


""" Hierarchical predictions """
if True:
    plt.figure(figsize=(6, 6), dpi=80)
    for t, (inp, pred) in enumerate(zip(vid_in, vid_p1)):
        if t < INTERPOLATION_START:
            plt.scatter(inp[0], inp[1], color="black")
            plt.scatter(pred[0], pred[1], color="blue")
        else:
            plt.scatter(inp[0], inp[1], color="grey")
            plt.scatter(pred[0], pred[1], color="green")
    plt.title("Prediction from hierarchical layer 1")
    plt.ylim(-1,1), plt.xlim(-1,1)
    plt.show()

if False:
    plt.figure(figsize=(6, 6), dpi=80)
    for t, (inp, pred) in enumerate(zip(vid_in, vid_p2)):
        if t < INTERPOLATION_START:
            plt.scatter(inp[0], inp[1], color="black")
            plt.scatter(pred[0], pred[1], color="blue")
        else:
            plt.scatter(inp[0], inp[1], color="grey")
            plt.scatter(pred[0], pred[1], color="green")
    plt.title("Prediction from hierarchical layer 2")
    plt.ylim(-1,1), plt.xlim(-1,1)
    plt.show()

    plt.figure(figsize=(6, 6), dpi=80)
    for t, (inp, pred) in enumerate(zip(vid_in, vid_p3)):
        if t < INTERPOLATION_START:
            plt.scatter(inp[0], inp[1], color="black")
            plt.scatter(pred[0], pred[1], color="blue")
        else:
            plt.scatter(inp[0], inp[1], color="grey")
            plt.scatter(pred[0], pred[1], color="green")
    plt.title("Prediction from hierarchical layer 1")
    plt.ylim(-1,1), plt.xlim(-1,1)
    plt.show()


""" Encoded generalized coordinates of input """
GC_TIME = False # align higher order generalized coordinates to a time axis
plt.figure(figsize=(6, 6), dpi=80)
for t, inp in enumerate(vid_in):
    if t < INTERPOLATION_START:
        l0 = plt.scatter(inp[0], inp[1],
                         color="black", label="Ground truth (burn in)")
    else:
        colors, lines = ["green", "yellow", "orange", "red", "brown", "purple"], []
        lines.append(plt.scatter(inp[0], inp[1], color="grey", label="Ground truth"))
        for gc in range(len(vid_cause_coords[t-INTERPOLATION_START])):
            select = 0 if GC_TIME else gc
            lines.append(plt.scatter(vid_cause_coords[t-INTERPOLATION_START][select][0],
                        vid_cause_coords[t-INTERPOLATION_START][gc][1],
                        color=colors[gc], label=f"Gen. coordinate {gc}"))
plt.title("Extrapolation of generalized coordinates\nprojected to discrete sequence")
plt.legend(handles=lines)
plt.ylim(-1,1), plt.xlim(-1,1)
plt.show()            
            
            
