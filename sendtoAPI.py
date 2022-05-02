import requests


def send_request(text):
    API_ENDPOINT  = ''
    
    CLOUD_NAME = 'http-sv-dut-udn-vn'
    
    API_KEY = '785855422285461'
    
    API_SECRET = 'JVJ5AmmxSIWseUELFaQ0RsZxEyA'
    
    
    data = {
        'text': text
    }
    
    r = requests.post(url = API_ENDPOINT, data = data)
    
    print("done")
    
