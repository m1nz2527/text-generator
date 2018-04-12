import os
import csv

os.chdir('./')

dictionary = {}
data = []

# Get data from https://www.kaggle.com/mousehead/songlyrics/data
with open('songdata.csv', 'r') as f:
    read = csv.reader(f, delimiter = ',')
    for row in read:
        data.append(row)
        artist = row[0]
        count = dictionary.get(artist)
        if count is None:
            count = 0
        dictionary[artist] = count + 1

artist_list = list(sorted(dictionary, key=dictionary.__getitem__))
artist_list.reverse()

artist_list_top100 = artist_list[:100]
artist_list_more90 = []
for artist in artist_list:
    if dictionary.get(artist) > 90:
        artist_list_more90.append(artist)

count1 = 0
count2 = 0
with open('top100artist.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter = ',')
    for row in data:
        if row[0] in artist_list_top100:
            count1 += 1
            writer.writerow(row)

print("Number of songs included in 'top100artist.csv' file: %d" %count1)

with open('morethan90.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    for row in data:
        if row[0] in artist_list_more90:
            count2+=1
            writer.writerow(row)

print("Number of songs included in 'morethan90.csv' file: %d, Number of artist: %d" 
        %(count2, len(artist_list_more90)))
