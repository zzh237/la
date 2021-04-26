import numpy as np 
import matplotlib.pyplot as plt
import torch 
import celluloid
from celluloid import Camera
from matplotlib import animation 
from experiments.minst import ConvNet
from mpl_toolkits.mplot3d import Axes3D

class contour:
    def __init__(self, model, train_dataset, criterion):
        self.train_dataset=train_dataset
        self.model = model
        self.criterion = criterion

        
    
    def contour_cost(self, batch_size, mp1, mp2):
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0,
                                                pin_memory=True)
        
        new_model = ConvNet()
        new_model.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            new_model.layer1[0].weight[15,0,4,2] = mp1
            new_model.layer1[0].weight[15,0,4,3] = mp2
            for i, (images, labels) in enumerate(train_loader):
                outputs = new_model(images)
                loss = self.criterion(outputs, labels)
                break 
        
        return loss 


    def create_contour_surface(self):
        # Set range of values for meshgrid: 
        m1s = np.linspace(-15, 17, 40)   
        m2s = np.linspace(-15, 18, 40)  
        self.M1, self.M2 = np.meshgrid(m1s, m2s) # create meshgrid 

        # Determine costs for each coordinate in meshgrid: 
        zs_100 = np.array([self.contour_cost(100 ,mp1, mp2)  
                            for mp1, mp2 in zip(np.ravel(self.M1), np.ravel(self.M2))])


        self.Z_100 = zs_100.reshape(self.M1.shape) # z-values for N=100


        # zs_10000 = np.array([costs(X_train[0:10000],y_train_oh[0:10000].T  
        #                             ,np.array([[mp1]]), np.array([[mp2]]),135)  
        #                     for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
        # Z_10000 = zs_10000.reshape(M1.shape) # z-values for N=10,000


        # # Plot loss landscapes: 
        # fig = plt.figure(figsize=(10,7.5)) # create figure
        # ax0 = fig.add_subplot(121, projection='3d' )
        # ax1 = fig.add_subplot(122, projection='3d' )

        # fontsize_=20 # set axis label fontsize
        # labelsize_=12 # set tick label size

        # # Customize subplots: 
        # ax0.view_init(elev=30, azim=-20)
        # ax0.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=9)
        # ax0.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=-5)
        # ax0.set_zlabel("costs", fontsize=fontsize_, labelpad=-30)
        # ax0.tick_params(axis='x', pad=5, which='major', labelsize=labelsize_)
        # ax0.tick_params(axis='y', pad=-5, which='major', labelsize=labelsize_)
        # ax0.tick_params(axis='z', pad=5, which='major', labelsize=labelsize_)
        # ax0.set_title('N:100',y=0.85,fontsize=15) # set title of subplot 

        # ax1.view_init(elev=30, azim=-30)
        # ax1.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=9)
        # ax1.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=-5)
        # ax1.set_zlabel("costs", fontsize=fontsize_, labelpad=-30)
        # ax1.tick_params(axis='y', pad=-5, which='major', labelsize=labelsize_)
        # ax1.tick_params(axis='x', pad=5, which='major', labelsize=labelsize_)
        # ax1.tick_params(axis='z', pad=5, which='major', labelsize=labelsize_)
        # ax1.set_title('N:10,000',y=0.85,fontsize=15)

        # # Surface plots of costs (= loss landscapes):  
        # ax0.plot_surface(M1, M2, Z_100, cmap='terrain', #surface plot
        #                             antialiased=True,cstride=1,rstride=1, alpha=0.75)
        # ax1.plot_surface(M1, M2, Z_10000, cmap='terrain', #surface plot
        #                             antialiased=True,cstride=1,rstride=1, alpha=0.75)
        # plt.tight_layout()
        # plt.show()

    def contour_anmiation(self, contour_res):
        costs, weights_a, weights_b = contour_res['cost'],contour_res['w_a'],contour_res['w_b']
        fig = plt.figure(figsize=(10,10)) # create figure
        ax = fig.add_subplot(111,projection='3d') 
        line_style=["dashed", "dashdot", "dotted"] #linestyles
        fontsize_=27 # set axis label fontsize
        labelsize_=17 # set tick label fontsize
        ax.view_init(elev=30, azim=-10)
        ax.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=17)
        ax.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=5)
        ax.set_zlabel("costs", fontsize=fontsize_, labelpad=-35)
        ax.tick_params(axis='x', pad=12, which='major', labelsize=labelsize_)
        ax.tick_params(axis='y', pad=0, which='major', labelsize=labelsize_)
        ax.tick_params(axis='z', pad=8, which='major', labelsize=labelsize_)
        ax.set_zlim(4.75,4.802) # set range for z-values in the plot

        # Define which epochs to plot:
        p1=list(np.arange(0,200,10))
        p2=list(np.arange(200,600,50))
        points_=p1+p2

        camera=Camera(fig) # create Camera object
        for i in points_:
            # Plot the three trajectories of gradient descent...
            #... each starting from its respective starting point
            #... and each with a unique linestyle:
            
            ax.plot(weights_a[0:i],weights_b[0:i],costs[0:i],
                        linestyle=line_style[0],linewidth=2,
                        color="black", label=str(i))
            ax.scatter(weights_a[i],weights_b[i],costs[i],
                        marker='o', s=15**2,
                    color="black", alpha=1.0)
            # Surface plot (= loss landscape):
            ax.plot_surface(self.M1, self.M2, self.Z_100, cmap='terrain', 
                                    antialiased=True,cstride=1,rstride=1, alpha=0.75)
            ax.legend([f'epochs: {i}'], loc=(0.25, 0.8),fontsize=17) # set position of legend
            plt.tight_layout() 
            camera.snap() # take snapshot after each iteration
            
            animation = camera.animate(interval = 5, # set delay between frames in milliseconds
                                    repeat = False,
                                    repeat_delay = 0)
            animation.save('gd_1.gif', writer = 'imagemagick', dpi=100)  # save animation 