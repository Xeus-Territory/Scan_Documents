import re
from num2words import num2words
from unicodedata import normalize


def vi_num2words(num):
    """
        Convert number to Vietnamese words
    """
    return num2words(num, lang='vi')


def convert_time_to_text(time_string):
    """
        Convert time to text
    """
    # Support only hh:mm format
    try:
        h, m = time_string.split(":")
        time_string = vi_num2words(int(h)) + " giờ " + \
            vi_num2words(int(m)) + " phút"
        return time_string
    except:
        return None


def replace_time(text):
    """
        Replace time to text
    """
    # Define regex to time hh:mm
    result = re.findall(r'\d{1,2}:\d{1,2}|', text)
    match_list = list(filter(lambda x: len(x), result))

    for match in match_list:
        if convert_time_to_text(match):
            text = text.replace(match, convert_time_to_text(match))
    return text


def replace_number(text):
    """
        Replace number to text
    """
    return re.sub('(?P<id>\d+)', lambda m: vi_num2words(int(m.group('id'))), text)


def normalize_text(text):
    """
        Normalize text with combine all of the above functions
    """
    text = normalize("NFC", text)
    text = text.lower()
    text = replace_time(text)
    text = replace_number(text)
    return text
