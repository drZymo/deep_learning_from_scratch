import matplotlib.pyplot as plt

loss_plot_fig = None
loss_plot_ax = None
train_losses = []
val_losses = []

def init_loss_plot():
    global train_losses, val_losses, loss_plot_fig, loss_plot_ax
    loss_plot_fig = plt.figure()
    loss_plot_ax = loss_plot_fig.add_subplot(111)
    train_losses = []
    val_losses = []
    plt.ion()
    loss_plot_ax.plot(train_losses, label='train loss')
    loss_plot_ax.plot(val_losses, label='validation loss')
    loss_plot_fig.legend()
    loss_plot_fig.show()
    loss_plot_fig.canvas.draw()

def finish_loss_plot():
    loss_plot_ax.clear()
    loss_plot_ax.plot(train_losses, label='train loss')
    loss_plot_ax.plot(val_losses, label='validation loss')
    loss_plot_fig.canvas.draw()
    
def add_loss_to_plot(train_loss, val_loss, redraw=False):
    global train_losses, val_losses, loss_plot_fig, loss_plot_ax
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if redraw:
        loss_plot_ax.clear()
        loss_plot_ax.plot(train_losses, label='train loss')
        loss_plot_ax.plot(val_losses, label='validation loss')
        loss_plot_fig.canvas.draw()