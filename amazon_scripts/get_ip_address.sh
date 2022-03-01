#!/bin/sh
awk '{ print $4 }' amazon_instances_info > nodeIPaddress
