#!/usr/bin/env python

import os
from os.path import exists
import requests
import shutil
import tarfile


gdrive_file_ids = {
    'Epoch_32_step_91046_mAP_0.8754_loss_2.2147_lr_3e-05.data-00000-of-00001' : '16rpSHpAzoKUpFDcZyjBv4ZynlNpMD5q4',
    'Epoch_32_step_91046_mAP_0.8754_loss_2.2147_lr_3e-05.index' : '1uyAxTgbzdfBxgJ-QDFWVIVrTW5-GY7kw',
    'Epoch_32_step_91046_mAP_0.8754_loss_2.2147_lr_3e-05.meta' : '1h77zEF1TVzBRKC3aUxnBiXYh0lK6oIQz'
}

    
    
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : file_id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : file_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':
    for file_name in gdrive_file_ids:
	    download_file_from_google_drive(gdrive_file_ids[file_name], './' + file_name)


    



