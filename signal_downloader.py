import urllib.request
import os
import sys


data_path = 'https://physionet.org/files/mitdb/1.0.0/'
urllib.request.urlretrieve(data_path+'RECORDS', 'data/RECORDS.txt')

records = []
download_first = -1
skip_if_exists = False


i = 0
for record in open('data/RECORDS.txt', 'r'):
    if 0 < download_first <= i:
        break
    records.append(record.replace('\n', ''))
    i += 1

file = open('data/RECORDS.txt', 'w')
for record in records:
    file.write(record+'\n')
file.close()


types = ['atr','dat','hea']


print("Downloading %s files %s" % ('all' if download_first < 0 else 'first %s' % download_first, 'and skipping existing ones' if skip_if_exists else 'and overwriting existing ones'))

downloaded_count = 0
skipped_count = 0
i = 0
for record in records:
    print("Downloading record", record)
    for type in types:
        fileName = "data/%s.%s" % (record, type)
        if not (skip_if_exists and os.path.exists(fileName)):
            open(fileName, 'a').close() # create empty file
            urllib.request.urlretrieve(data_path + record + '.' + type, fileName)
            downloaded_count += 1
        else:
            skipped_count += 1

print("Downloaded %d files, skipped %d files" % (downloaded_count, skipped_count))

