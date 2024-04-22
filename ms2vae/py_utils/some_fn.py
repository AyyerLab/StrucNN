def quaternion_angle(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot_product = np.dot(q1, q2)
    angle = np.arccos(2 * dot_product**2 - 1)
    return angle


all_angle_between_epochs = []
with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/vae_ori/vae_train_data/vae_r78_round1_epoch50.h5', 'r') as f:
    orient_1 = f['coors'][:]
for i in np.arange(100, 750, 50):
    print(i)
    with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/vae_ori/vae_train_data/vae_r78_round1_epoch%.1d.h5' % (i), 'r') as f:
        orient_2 = f['coors'][:]
    angle_per_epoch = []
    for j in range(5376):
        angle = quaternion_angle(orient_1[j], orient_2[j])
        angle_per_epoch.append(angle)
    all_angle_between_epochs.append(angle_per_epoch)
all_angle_between_epochs = np.array(all_angle_between_epochs)


#             OR

all_angle_between_epochs = []
for i in np.arange(50, 750, 50):
    if i == 700: break
    with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/vae_ori/vae_train_data/vae_r78_round1_epoch%.1d.h5'%i, 'r') as f:
        orient_1 = f['coors'][:]

    with h5py.File('/u/mallabhi/StrucNN/ms2vae/output/vae_ori/vae_train_data/vae_r78_round1_epoch%.1d.h5' % (i+50), 'r') as f:
        orient_2 = f['coors'][:]
    angle_per_epoch = []
    for j in range(5376):
        angle = quaternion_angle(orient_1[j], orient_2[j])
        angle_per_epoch.append(angle)
    all_angle_between_epochs.append(angle_per_epoch)
all_angle_between_epochs = np.array(all_angle_between_epochs)



all_rmsd = []
for i in range(len(all_angle_between_epochs)-1):
    rmsd = np.sqrt(np.mean(np.square(all_angle_between_epochs[i])))
    all_rmsd.append(rmsd)
all_rmsd = np.array(all_rmsd)
#                      OR
all_rmsd = []
for i in range(len(all_angle_between_epochs)-1):
    mask = ~np.isnan(all_angle_between_epochs[i])
    all_angle_between_epochs_ = all_angle_between_epochs[i][mask]
    rmsd = np.sqrt(np.mean(np.square(all_angle_between_epochs_)))
    all_rmsd.append(rmsd)
all_rmsd = np.array(all_rmsd)

max_theta = []
for i in range(len(all_angle_between_epochs)-1):
    theta = all_angle_between_epochs.max()
    max_theta.append(theta)
max_theta = np.array(max_theta)

