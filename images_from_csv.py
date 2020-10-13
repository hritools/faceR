import os
import requests

with open('raw.csv') as f:
    lis=[line.split(',') for line in f]
    for i, person in enumerate(lis):
        person[0] = person[0].replace(' ', '_')
        person[1] = person[1].strip('\n')
        print("Will create dir {0}, and store image from {1}".format(person[0], person[1]))

        if not os.path.exists('raw/'+person[0]):
            os.makedirs('raw/'+person[0])

        with open('raw/{0}/{1}'.format(person[0], person[1].split('/')[-1]), 'wb') as handle:
            response = requests.get(person[1], stream=True)

            if not response.ok:
                print(response)

            for block in response.iter_content(1024):
                if not block:
                    break

                handle.write(block)
