import json
import simplejson
from pprint import pprint

data = json.load(open('data.json'))
dataB = json.load(open('dataB.json'))

data_small = {}
data_small['tsnewavenet00'] = data['tsnewavenet00'] 
data_small['umapwavenet00'] = data['umapwavenet00'] 
data_small['tsnemfcc00'] = data['tsnemfcc00']
data_small['umapmfcc00'] = data['umapmfcc00']
data_small['pcamfcc'] = data['pcamfcc']
data_small['pcawavenet'] = data['pcawavenet']
data_small['filenames'] = data['filenames']



for key,value in data.iteritems():
    print key
print 
print
for key,value in dataB.iteritems():
    print key

simplejson.dump(data_small, open('data_Small.json','w'), encoding='utf-8', ignore_nan=True)

print data[0][0]


#pprint(data['filenames'])
#pprint(dataB['umapmodelB00'][0])