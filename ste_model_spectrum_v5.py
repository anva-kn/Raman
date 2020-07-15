# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:20:49 2020

@author: Stefano rini
"""



# CHANGE THIS FUNCTION TO HAVE A THE MEAN AROUND THE PEAK BOUNDARY

def find_peak_supp(peak,l_pos,r_pos, Y,peak_tol=0.9):
# win_len_mul= how much do we extend the window length 
# smooth_win= how much do we smooth the function
    res=Y.shape[0]
    Y_smooth=Y
    
    
    if l_pos==r_pos:
        y_data=Y_smooth[l_pos]
        Y_max=y_data
        Y_max_idx=l_pos
        Y_min=Y_max
    else:
        y_data=Y_smooth[l_pos:r_pos]
        Y_max=np.max(y_data)
        Y_max_idx=l_pos+np.argmax(Y[l_pos:r_pos])        
        Y_min=np.mean(np.sort(y_data)[0:int(y_data.size/10)+1])
        
    l_pos_temp=max(Y_max_idx-1,0)
    r_pos_temp=min(Y_max_idx+1,res-2)
        
    # l_pos_temp=Y_max_idx-1
    # r_pos_temp=Y_max_idx+1
        
    while  Y_smooth[l_pos_temp-1]*peak_tol<=Y_smooth[l_pos_temp]  and l_pos_temp>0:
        l_pos_temp=l_pos_temp-1
    
    while  Y_smooth[r_pos_temp+1]*peak_tol<=Y_smooth[r_pos_temp] and r_pos_temp<res-2: 
        r_pos_temp=r_pos_temp+1
        
    return [l_pos_temp,r_pos_temp]



def find_mul_peak_supp(int_idx, y_data_win, mean_level_win, discount=0.9):
    
    Y_smooth=y_data_win-mean_level_win
    # bring positive
    Y_smooth=Y_smooth-min(Y_smooth)
    res=Y_smooth.shape[0]

    l_pos=int_idx[0]
    r_pos=int_idx[1]
    
    l_pos_temp=int(np.max([l_pos-1,0]))
    r_pos_temp=int(np.min([r_pos+1,res-1]))
            

    if l_pos_temp>0:
        win_left=np.mean(Y_smooth[:l_pos_temp])
        
        while win_left<=Y_smooth[l_pos_temp]  and  Y_smooth[l_pos_temp-1]>0  and l_pos_temp>0:        
            win_left=np.mean(Y_smooth[:l_pos_temp])
            l_pos_temp=l_pos_temp-1
                
    if r_pos_temp<res-1:                        
        win_right=np.mean(Y_smooth[r_pos_temp:])
        
        while  win_right<=Y_smooth[r_pos_temp] and Y_smooth[r_pos_temp+1]>0 and r_pos_temp<res-2:     
            win_right=np.mean(Y_smooth[r_pos_temp: ])
            r_pos_temp=r_pos_temp+1

           
    l_pos_temp=l_pos_temp-1
    r_pos_temp=r_pos_temp+1
    
    l_pos=max(l_pos_temp-1,0)
    r_pos=min(r_pos_temp+1,res-1)
          
    if False:
        plt.plot(Y_smooth)
        plt.plot(range(l_pos,r_pos),Y_smooth[l_pos:r_pos],'y--')
        plt.plot(range(l_pos_temp,r_pos_temp),Y_smooth[l_pos_temp:r_pos_temp],'r')
                       
    return [int(l_pos),int(r_pos)]

