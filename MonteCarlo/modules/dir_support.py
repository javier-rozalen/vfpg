# -*- coding: utf-8 -*-
"""
Created on Wed May 11 23:52:11 2022

@author: javir
"""
import os

def dir_support(nested_dirs,path_to_this_module):
    """
    Directories support: ensures that the (nested) directories given via the 
    input list do exist, creating them if necessary. 
    
    Parameters
    ----------
    nested_dirs : list
        Contains all nested directories in order.

    Returns
    -------
    None.

    """
    
    for i in range(len(nested_dirs)):
        potential_dir = '/'.join(nested_dirs[:i+1])
        if not os.path.exists(potential_dir):
            os.makedirs(potential_dir)
            print(f'Creating directory {potential_dir}...')
