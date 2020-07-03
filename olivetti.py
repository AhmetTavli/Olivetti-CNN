# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from os import mkdir, listdir, getcwd
from os.path import join, exists
from cv2 import imwrite
from shutil import rmtree
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms import Grayscale
from torch.utils.data import DataLoader
from torch.nn import Module, Conv2d, Dropout2d, Linear
from torch.nn.functional import relu, max_pool2d
from torch.nn.functional import log_softmax, nll_loss
from torch import flatten, manual_seed, device, save
from torch import no_grad
from torch.optim.lr_scheduler import StepLR
from torch.optim.adadelta import Adadelta
from torch.cuda import is_available
from argparse import ArgumentParser
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from imutils.paths import list_images
from matplotlib.pyplot import plot, legend, show, savefig
from matplotlib.pyplot import suptitle, subplots


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=32,
                            kernel_size=3, stride=1)
        self.conv2 = Conv2d(in_channels=32, out_channels=64,
                            kernel_size=3, stride=1)
        self.dropout1 = Dropout2d(p=0.5)
        self.fc1 = Linear(in_features=12544, out_features=128)
        self.fc2 = Linear(in_features=128, out_features=40)

    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = max_pool2d(x, 2)

        x = self.conv2(x)
        x = relu(x)
        x = max_pool2d(x, 2)

        x = flatten(x, 1)

        x = self.fc1(x)
        x = relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        out = log_softmax(x, dim=1)
        return out


def train(argument_object, model, dev, train_loader,
          optimizer, epoch):
    """
    Args:
         argument_object (Namespace): Network params
         model (Net): CNN model
         dev (device): If CUDA, enables GPU, CPU otherwise
         train_loader (DataLoader): Train dataset
         optimizer (Adadelta): Adadelta object
         epoch (int): Iteration number
    """
    model.train()
    run_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(dev), target.to(dev)
        optimizer.zero_grad()
        output = model(data)
        loss = nll_loss(output, target)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        if batch_idx % argument_object.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Loss: {:.6f}'.format(epoch,
                                        batch_idx * len(data),
                                        len(train_loader.dataset),
                                        100. * batch_idx /
                                        len(train_loader),
                                        loss.item()))
    return run_loss


def test(model, dev, test_loader):
    """
    Args:
         model (Net): CNN model
         dev (device): If CUDA, enables GPU, CPU otherwise
         test_loader (DataLoader): Test dataset
    """
    model.eval()
    test_loss = 0
    correct = 0
    with no_grad():
        for data, target in test_loader:
            data, target = data.to(dev), target.to(dev)
            output = model(data)
            test_loss += nll_loss(output, target,
                                  reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                               len(test_loader.dataset),
                                               100. * correct /
                                               len(test_loader.dataset)))
    return test_loss


def check_save(size, to_dir):
    """Assert save_image method works properly.

    Args:
        size (int): Size of (train-validation-test) set
        to_dir (string): Either train, validation or test directory.
    """
    assert (size > 0)
    check_size = 0
    entries = listdir(path=to_dir)
    for entry in entries:
        images = listdir(path=join(to_dir, entry))
        check_size += len(images)
    assert (check_size == size)


def save_image(size, label, to_dir, x):
    """
    Args:
        size (int): Size of (train-validation-test) set
        label (string): Class names
        to_dir (string): Either train, validation or test directory.
        x (ndarray): image data
    """
    current = str(label[0])
    count = 0
    for i in range(size):
        if current != str(label[i]):
            current = str(label[i])
        if len(str(current)) == 1:
            current = '0' + str(current)
        label_dir = join(to_dir, current)
        if not exists(path=label_dir):
            mkdir(path=label_dir)
        name = join(label_dir, str(str(label[i]) + '_' +
                                   str(count) + '.png'))
        imwrite(filename=name, img=x[i] * 255)
        count += 1

    check_save(size=size, to_dir=to_dir)


def generate_and_save(data_gen, set_path, set_dir, gen_num,
                      save_format=".png", save_prefix="gen_image"):
    """Populate images using data augmentation.

    Args:
        data_gen (ImageDataGenerator): Generator object
        set_path (str):
        set_dir (list):
        gen_num (int):
        save_format (str):
        save_prefix (str):
    """
    for label in set_dir:
        if label != ".DS_Store":  # Mac-os specific problem
            path = join(set_path, label)
            images = listdir(path=path)

            for img in images:
                if img != '.DS_Store':  # Mac-os specific problem
                    count_img = 1
                    img_path = join(path, img)
                    loaded_img = load_img(path=img_path,
                                          color_mode="grayscale")
                    array_img = img_to_array(img=loaded_img)
                    current_img = array_img.reshape((1,) + array_img.shape)

                    for _ in data_gen.flow(x=current_img, batch_size=1,
                                           save_to_dir=path,
                                           save_prefix=save_prefix,
                                           save_format=save_format):
                        count_img += 1
                        if count_img > gen_num:
                            break


def create_folder(write_to_file):
    """Create train and test folders under data folder.
       If data folder is existed, deletes and recreates it.

    Args:
        write_to_file (bool, optional): If true, save all images
            to the data folder.

    Returns:
        tuple: (train, test) both are the directory paths.
    """
    dir_data = "data"
    dir_train = join(dir_data, "train")
    dir_test = join(dir_data, "test")
    if write_to_file:
        if exists(path=dir_data):
            rmtree(path=dir_data)
        mkdir(path=dir_data)
        mkdir(path=dir_train)
        mkdir(path=dir_test)

    print("All sets (train-test) are available under"
          " the data folder:\n\n{}\n".format(getcwd()))

    return dir_train, dir_test


def design_data(x, y, test_size):
    """
    Args:
        x (ndarray): images
        y (ndarray): target
        test_size (float): Dividing dataset based on the ratio,
            remaining part will be the test set.

    Returns:
        training data, training target, test data, test target
    """
    separated_data = train_test_split(x, y, test_size=test_size,
                                      random_state=42)
    x_train = separated_data[0]
    x_test = separated_data[1]
    y_train = separated_data[2]
    y_test = separated_data[3]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, x_test, y_train, y_test


def get_images(test_size):
    """
    Args:
        test_size (float): Dividing dataset based on the ratio,
            remaining part will be the test set.
    Returns:
         tuple: (train and test images)
    """
    olivetti = fetch_olivetti_faces()
    images = olivetti.images
    target = olivetti.target
    train_image, test_image, _, _ = train_test_split(images, target,
                                                     test_size=test_size,
                                                     random_state=42)
    return train_image, test_image


def split_data(test_size, generate_data, write_to_file=True):
    """
    Args:
        test_size (float): Dividing dataset based on the ratio,
            remaining part will be the test set.
        generate_data (bool, optional): If true, generate and
            save all images to the olivetti folder.
        write_to_file (bool, optional): If true, save all images
            to the data folder.

    Returns:
        dict: (training data, training target, test data, test target)
    """
    x, y = fetch_olivetti_faces(return_X_y=True)
    x_train, x_test, y_train, y_test = design_data(x=x, y=y,
                                                   test_size=test_size)

    dir_train, dir_test = create_folder(write_to_file=write_to_file)
    size_train = x_train.shape[0]
    size_test = x_test.shape[0]
    train_folder_size = 0
    test_folder_size = 0
    img_row, img_col = 64, 64
    train_images, test_images = get_images(test_size=test_size)
    if write_to_file:
        save_image(size=size_train, label=y_train, to_dir=dir_train,
                   x=train_images)
        save_image(size=size_test, label=y_test, to_dir=dir_test,
                   x=test_images)
        train_folder_size = len(list(list_images(basePath=dir_train)))
        test_folder_size = len(list(list_images(basePath=dir_test)))
        print("\nTrain folder images: {}".format(train_folder_size))
        print("Test folder images: {}".format(test_folder_size))
    if generate_data:
        train_num = int(60000 / train_folder_size)
        test_num = int(10000 / test_folder_size)
        train_data_gen = ImageDataGenerator(rescale=1./255,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True)
        test_data_gen = ImageDataGenerator(rescale=1./255)
        example_img = train_images.reshape(train_images.shape[0],
                                           img_row, img_col, 1)[0]
        example_img = example_img.reshape((1,) + example_img.shape)
        count = 0
        gen_number = 10  # generate 10 samples from the example image
        gen_images = []
        gen_labels = []
        for batch in train_data_gen.flow(x=example_img, batch_size=1):
            batch_reshaped = batch.reshape(batch.shape[1], batch.shape[2])
            gen_images.append(batch_reshaped)
            gen_labels.append(y_train[0])
            count += 1
            if count > gen_number:
                break
        t_gen = "Generated Train Samples"
        f_gen = "gen_train_samples.png"
        display_generated_samples(2, 5, x=gen_images, y=gen_labels,
                                  t=t_gen, title="Id:{}", f_name=f_gen)
        print("\nTrain generation begins..")
        generate_and_save(data_gen=train_data_gen, set_path=dir_train,
                          set_dir=listdir(path=dir_train), gen_num=train_num)
        train_generated = len(list(list_images(basePath=dir_train)))
        print("\nTrain generation end. Generated Train images: {}".
              format(train_generated))
        print("\nTest generation begins..")
        generate_and_save(data_gen=test_data_gen, set_path=dir_test,
                          set_dir=listdir(path=dir_test), gen_num=test_num)
        test_generated = len(list(list_images(basePath=dir_test)))
        print("\nTest generation end. Generated Test images: {}".
              format(test_generated))

    transform = Compose(transforms=[Grayscale(num_output_channels=1),
                                    RandomHorizontalFlip(),
                                    ToTensor()])

    train_dataset = ImageFolder(root=dir_train, transform=transform)
    test_dataset = ImageFolder(root=dir_test, transform=transform)
    data = {'train_dataset': train_dataset, 'training_label': y_train,
            'test_dataset': test_dataset, 'test_label': y_test}
    return data


def plot_loss(train_loss, test_loss):
    r"""Plot train and the test loss
    Args:
        train_loss (list): Training loss during epoch
        test_loss (list): Test loss during epoch
    """
    plot(train_loss, label='Training loss')
    plot(test_loss, label='Test los')
    legend(frameon=False)
    savefig(fname="olivetti_loss.png", dpi=300)
    show()


def display_generated_samples(n_row, n_col, x, y, t, title="Id:{}",
                              fig_size=(6, 3), dpi=300,
                              f_name="default.png"):
    """
    Args:
        n_row (int): Row number
        n_col (int): Column number
        x (list): generated images
        y (list): labels
        t (str): Graph title
        title (str): Id title
        fig_size (tuple): figure size
        dpi (int): dots per inch
        f_name (str): file name
    """
    fig, ax = subplots(nrows=n_row, ncols=n_col, figsize=fig_size, dpi=dpi)
    ax = ax.flatten()

    sample_num = n_row * n_col

    for i in range(sample_num):
        ax[i].imshow(X=x[i], cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(title.format(y[i]))
    suptitle(t=t)
    savefig(f_name)


def arguments(train_batch_size=64, test_batch_size=32, epochs=5,
              learning_rate=1.0, test_size=0.35, gamma=0.7,
              no_cuda=True, seed=1, log_interval=10, save_model=True,
              write_to_file=True, generate_data=True,
              log_dir='runs/olivetti_experiment'):
    """
    Args:
        train_batch_size (int): Input batch size for training
        test_batch_size (int): Input batch size for testing
        epochs (int): Number of episodes for training
        learning_rate (float): Step size at each iteration
        test_size (float): Split ratio
        gamma (float): Learning rate step
        no_cuda (bool): If true, disables CUDA
        seed (int): Value of the random seed
        log_interval (int): Step to save batches
        save_model (bool): If true, saves model
        write_to_file (bool): If true, creates train/test dir
        generate_data (bool): If true, populate data
        log_dir (str): Tensorboard run location
    Returns:
         argument object
    """
    parser = ArgumentParser(description="Olivetti Example")
    parser.add_argument('--train-batch-size', type=int,
                        default=train_batch_size, metavar='N',
                        help='input batch size for train'
                             ' (default: {})'.format(train_batch_size))
    parser.add_argument('--test-batch-size', type=int,
                        default=test_batch_size, metavar='N',
                        help='input batch size for test'
                             ' (default: {})'.format(test_batch_size))
    parser.add_argument('--epochs', type=int, default=epochs,
                        metavar='N',
                        help='number of epochs to train'
                             ' (default: {})'.format(epochs))
    parser.add_argument('--lr', type=float, default=learning_rate,
                        metavar='LR',
                        help='learning rate '
                             '(default: {})'.format(learning_rate))
    parser.add_argument('--test-size', type=float, default=test_size,
                        metavar='N',
                        help='test size split ratio '
                             '(default: {})'.format(test_size))
    parser.add_argument('--gamma', type=float, default=gamma, metavar='M',
                        help='Learning rate step gamma '
                             '(default: {})'.format(gamma))
    parser.add_argument('--no-cuda', action='store_true', default=no_cuda,
                        help='disables CUDA training '
                             '(default: {})'.format(no_cuda))
    parser.add_argument('--seed', type=int, default=seed, metavar='S',
                        help='random seed (default: {})'.format(seed))
    parser.add_argument('--log-interval', type=int, default=log_interval,
                        metavar='N',
                        help='how many batches to wait '
                             'before logging training status '
                             '(default: {})'.format(log_interval))
    parser.add_argument('--save-model', action='store_true', default=save_model,
                        help='For Saving the current Model '
                             '(default: {})'.format(save_model))
    parser.add_argument('--write-to-file', type=bool, metavar='F',
                        default=write_to_file,
                        help='Split dataset into the train and test directories')
    parser.add_argument('--generate-data', type=bool, metavar='G',
                        default=generate_data,
                        help='Populate data similar to the MNIST')
    parser.add_argument('--log-dir', action='store_true', default=log_dir,
                        help='Tensorboard run location')
    argument_object = parser.parse_args()
    return argument_object


def main():
    args = arguments()
    manual_seed(seed=args.seed)
    use_cuda = not args.no_cuda and is_available()
    dev = device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    data = split_data(test_size=0.35, generate_data=args.generate_data,
                      write_to_file=args.write_to_file)
    train_dataset = data['train_dataset']
    test_dataset = data['test_dataset']
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.test_batch_size,
                             shuffle=True, **kwargs)
    model = Net().to(device=dev)
    optimizer = Adadelta(params=model.parameters(), lr=args.lr, rho=0.9,
                         eps=1e-6, weight_decay=0)
    scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=args.gamma)
    train_los, test_los = [], []
    for epoch in range(1, args.epochs + 1):
        tr_los = train(argument_object=args, model=model, dev=dev,
                       train_loader=train_loader, optimizer=optimizer,
                       epoch=epoch)
        te_los = test(model=model, dev=dev, test_loader=test_loader)
        scheduler.step(epoch=epoch)
        train_los.append(tr_los/len(train_loader))
        test_los.append(te_los)
    if args.save_model:
        save(obj=model.state_dict(), f="olivetti_cnn.h5")
    if args.epochs > 1:
        plot_loss(train_loss=train_los, test_loss=test_los)


if __name__ == '__main__':
    main()
