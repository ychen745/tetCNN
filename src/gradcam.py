def saliency_map(input_grads):
    print('saliency_map')
    node_saliency_map = []
    for n in range(input_grads.shape[0]):
        node_grads = input_grads[n, :]
        node_saliency = torch.norm(F.relu(node_grads)).item()
        node_saliency_map.append(node_saliency)
    return node_saliency_map

def grad_cam(final_conv_acts, final_conv_grads):
    print('grad_cam')
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0)
    for n in range(final_conv_acts.shape[0]):
        node_heat = F.relu(alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    return node_heat_map

optimizer.zero_grad()
out,h = model(X)
model.final_conv_acts.retain_grad()
loss = criterion(out, X.y)  # Compute the loss.
#print(loss)
loss.backward()  # Derive gradients.

grad_cam_weights = grad_cam(model.final_conv_acts, model.final_conv_grads)
scaled_grad_cam_weights = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(grad_cam_weights).reshape(-1, 1))

from torch_geometric.nn import  knn_interpolate
tmp = torch.tensor(scaled_grad_cam_weights)
print(tmp.view(-1,1).cuda())
h.x = tmp.view(-1,1)
h = h.to(device)
x = knn_interpolate(h.x, h.pos, X.pos)
np.savetxt(os.path.join(os.path.abspath('.'),'test'),scaled_saliency_map_weights, fmt='%.6e')