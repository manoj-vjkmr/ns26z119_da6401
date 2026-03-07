from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
import wandb

wandb.init(project="da6401", name="Data Exploration and Class Distribution")

table = wandb.Table(columns=["Class", "Images"])
for cls in range(10):
    imgs = x_train[y_train == cls][:5] 
    for img in imgs:
        table.add_data(cls, wandb.Image(img))

wandb.log({"Sample Images per Class": table})