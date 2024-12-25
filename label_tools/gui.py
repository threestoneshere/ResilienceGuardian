import matplotlib
matplotlib.use('Agg')
from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt


from PIL import Image,ImageTk
import os
import pandas as pd
import numpy as np
import json


def normalize_time_series(values):
    min = np.min(values)
    max = np.max(values)
    amplitude = max - min
    if amplitude == 0:
        return np.ones(len(values))
    values_normalized = (values - min) / amplitude
    return values_normalized


def comparative_plots(dir_v1, dir_v2, save_path, normalize=True):
    for file in os.listdir(dir_v1):
        try:
            data_v1 = pd.read_csv(os.path.join(dir_v1, file))['value']
        except:
            print(file)
            continue
        data_v2 = pd.read_csv(os.path.join(dir_v2, file))['value']
        if normalize:
            data_v1 = normalize_time_series(data_v1)
            data_v2 = normalize_time_series(data_v2)
        comparative_plot(data_v1, data_v2, save_path, file[:-4])


def comparative_plot(data_v1, data_v2, save_path, name):     
    plt.rcParams['figure.figsize'] = (20.0, 8.0)
    plt.rcParams['savefig.dpi'] = 300 
    plt.rcParams['figure.dpi'] = 300
    assert len(data_v1) == len(data_v2)
    x = np.arange(0, len(data_v1))
    plt.subplot(211)
    plt.ylim(ymax=1.2)
    plt.ylim(ymin=-0.2)
    x_vertical = [92 + i * 240 for i in range(3)] + [148 + i * 240 for i in range(3)]
    for x_v in x_vertical:
        plt.axvline(x_v, color=(0, 0, 0), linewidth=0.5)
    plt.plot(x, data_v1, color=(1, 0, 0), zorder=1, linewidth=0.5, label='v1')
    plt.plot(x, data_v2, color=(0, 0, 1), zorder=1, linewidth=0.5, label='v2')
    plt.legend()
    plt.subplot(212)
    smoothing_window = int(len(data_v2) * 0.02)
    difference = np.array(pd.DataFrame(data_v2 - data_v1).ewm(span=smoothing_window).mean().values.flatten())
    max = np.max(difference) + 0.1
    min = np.min(difference) + 0.1
    plt.ylim(ymax=0.6 if max < 0.6 else max)
    plt.ylim(ymin=-0.6 if min > -0.6 else min)
    for x_v in x_vertical:
        plt.axvline(x_v, color=(1, 0, 0), linewidth=0.5)
    plt.plot(x, difference, color=(0, 0, 0), zorder=1, linewidth=0.5, label='difference')
    plt.legend()
    plt.savefig(os.path.join(save_path, '{}.png'.format(name)))
    plt.close()


def prepare_before_label(data_dir, plot_dir):
    for exp in os.listdir(data_dir):
        exp_dir = os.path.join(data_dir, exp)
        for target in os.listdir(exp_dir):
            dir_v1 = os.path.join(exp_dir, target, 'v1')
            dir_v2 = os.path.join(exp_dir, target, 'v2')
            fig_save_dir = os.path.join(plot_dir, exp, target)
            if os.path.isdir(fig_save_dir):
                print(f'Plots of {exp} {target} have been generated.')
                continue
            else:
                os.makedirs(fig_save_dir)
                comparative_plots(dir_v1, dir_v2, fig_save_dir)


def begin_label_tool(data_dir, plot_dir, info_file):
    with open(info_file) as f:
        exps = json.load(f)
        
    global index
    global current_fig
    global fig_labels
    def set_img_by_index():
        global index
        global current_fig
        current_fig = figs[index].split(delimiter)[-1][:-4]
        if current_fig in fig_labels.keys():
            label2_text.config(text=f'label: {fig_labels[current_fig]}')
        else:
            label2_text.config(text=f'not labelled')
        img = Image.open(figs[index])
        img = img.resize((1200, 550))
        img = ImageTk.PhotoImage(img)
        label.config(image=img)
        label_text.config(text=f'{index + 1}/{length}: {current_fig}')
        label.image = img
    
    def before():
        global index
        global current_fig
        index -= 1
        if index < 0:
            index = length - 1
        set_img_by_index()
        
    def next():
        global index
        global current_fig
        index += 1
        if index > length - 1:
            index = 0
        set_img_by_index()
           
        
    def positive():
        global current_fig
        global fig_labels
        fig_labels[current_fig] = 1
        label2_text.config(text=f'label: {fig_labels[current_fig]}')
        
    def negative():
        global current_fig
        global fig_labels
        fig_labels[current_fig] = 0
        label2_text.config(text=f'label: {fig_labels[current_fig]}')
    
    def output():
        global index
        global current_fig
        global fig_labels
        if len(fig_labels.keys()) == length:
            with open(os.path.join(label_save_dir, 'labels.json'), 'w') as f:
                json.dump(fig_labels, f)
                messagebox.showinfo(title=f'Great!', message='Labels have been output successfully!')
        else:
            for i in range(length):
                fig = figs[i].split(delimiter)[-1][:-4]
                if fig not in fig_labels.keys():
                    index = i
                    set_img_by_index()
                    break
    
    
    def more_info():
        info = json.dumps(exps[exp])
        messagebox.showinfo(title=f'{exp} info', message=info)
    
    
    for exp in os.listdir(data_dir):
        exp_dir = os.path.join(data_dir, exp)
        for target in os.listdir(exp_dir):
            target_dir = os.path.join(exp_dir, target)
            figs_dir = os.path.join(plot_dir, exp, target)
            if 'labels.json' not in os.listdir(target_dir):
                fig_labels = {}
                figs = [os.path.join(figs_dir, fig) for fig in os.listdir(figs_dir)]
                label_gui=Tk()
                label_gui.title(f'label-{exp}-{target}')
                index = 0
                current_fig = figs[index].split(delimiter)[-1][:-4]
                label_save_dir = target_dir
                length = len(figs)
                box1=Frame(label_gui, width=1300, height=50, borderwidth=5)
                box1.grid(row=0, column=0)
                label_text=Label(box1)
                label_text.grid(row=0, column=0)
                btmoreinfo = Button(box1, text="more info", command=more_info)
                btmoreinfo.grid(row=0, column=1)
                box=Frame(label_gui, width=1300, height=700, borderwidth=5)
                box.grid(row=1, column=0)
                label=Label(box, width=1300, height=600)
                label.grid(row=1, columnspan=2)
                btnbefore = Button(box, text="before", command=before)
                btnnext = Button(box, text="next", command=next)
                btnbefore.grid(row=2, column=0)
                btnnext.grid(row=2, column=1)
                btpostive = Button(box, text="positive", command=positive)
                btnegative = Button(box, text="negative", command=negative)
                box2 = Frame(label_gui, width=1300, height=20, borderwidth=5)
                box2.grid(row=3, column=0)
                label2_text = Label(box2)
                label2_text.grid(row=3, columnspan=2)
                btpostive.grid(row=4, column=0)
                btnegative.grid(row=4, column=1)
                btoutput = Button(box, text="output", command=output)
                btoutput.grid(row=5, columnspan=2)
                set_img_by_index()
                label_gui.mainloop()
            else:
                print(f'The data of {exp}-{target} has been labeled.')
                continue
        

    
delimiter = '\\'
index = -1
current_fig = None
fig_labels = None
data_dir = 'comparative_situation'
plot_dir = 'comparative_plots'
info_file = 'disruptions.json'
print(f'Generate comparative plots before labeling...')
prepare_before_label(data_dir, plot_dir)
print('All plots have been generated.')
begin_label_tool(data_dir, plot_dir, info_file)




            
            
            

