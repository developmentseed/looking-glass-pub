"""
Populate an SQS queue from a list of geojson boundaries

@author:developmentseed

pred_populate_sqs.py
"""
import os
import subprocess as subp

import boto3
from tqdm import tqdm

from config import sqs_params as SP

# Initialize the SQS queue
sqs = boto3.resource('sqs')
queue = sqs.create_queue(QueueName=SP['queue_name'],
                         Attributes={'DelaySeconds': '5'})

# Iterate over geojsons
for geojson_fpath in SP['geojson_fpaths']:
    print('\nProcessing geojson: {}'.format(geojson_fpath))

    # TODO: Might need to route this to a file if output is too big for RAM
    # Run geojson through GeoDex, get tiles
    std_result = subp.run(['geodex', geojson_fpath, str(SP['zoom']),
                           '--output-format', '{z} {x} {y}'], stdout=subp.PIPE)

    tile_inds = std_result.stdout.decode('utf-8').split('\n')
    tile_inds = [ind.split(' ') for ind in tile_inds if len(ind)]
    print('Found {} tiles, pushing to SQS Queue: {}'.format(len(tile_inds),
                                                            SP['queue_name']))

    # Iterate through tile list, push to SQS
    for ti in tqdm(tile_inds):
        msg_atrs = dict(geojson_fpath=dict(StringValue=geojson_fpath, DataType='String'),
                        tile_info=dict(StringValue=str(ti), DataType='String'))
        msg_body = SP['message_body'].format(z=ti[0], x=ti[1], y=ti[2],
                                             token=os.environ['VIVID_ACCESS_TOKEN'])

        # Create the new message using attributes
        response = queue.send_message(MessageBody=msg_body,
                                      MessageAttributes=msg_atrs)
        if response.get('Failed'):
            print(response.get('Failed'))

        #print(response.get('MessageId'))
        #print(response.get('MD5OfMessageBody'))
