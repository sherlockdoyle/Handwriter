import random
import uno


avg_mistake_dist = 20  # average number of words between two misspelled words
border_color = 0xff0000  # border color around misspelled word
single_char_font = 'Hand1'  # font name of single letter words

end_marker = 'INTRODUCEMISTAKEENDMARKER'
# It seems that there's no easy way to get a cursor spanning a selection. So this word is temporarily inserted at the
# end of the selection.


def introduceMistakes():
    doc = XSCRIPTCONTEXT.getDocument()
    selections = doc.getCurrentController().getSelection()

    for i in range(selections.Count):
        block = selections.getByIndex(i)
        block.Text.insertString(block.getEnd(), ' '+end_marker+' ', True)  # insert the marker

        cursor = block.Text.createTextCursorByRange(block)
        if cursor.isStartOfWord():
            cursor.goLeft(0, False)
        else:
            cursor.gotoNextWord(False)

        while True:
            cursor.gotoEndOfWord(True)
            if cursor.String == end_marker:
                # remove the marker and break out, since reached end of selection
                cursor.goRight(1, True)
                cursor.String = ''
                cursor.goLeft(1, True)
                cursor.String = ''
                break

            # comment out the following line to disable changing the font for uppercase letters
            fix_upper(cursor)

            if random.randint(0, avg_mistake_dist) == 0:
                word = cursor.String
                mword = wrong(word)

                if not word.startswith(mword):
                    cursor.String = mword+' '+word
                    mcur = cursor.Text.createTextCursorByRange(cursor)
                    mcur.goLeft(0, False)
                    mcur.goRight(len(mword), True)

                    border = uno.createUnoStruct('com.sun.star.table.BorderLine2',
                                                 Color=border_color,
                                                 OuterLineWidth=2,
                                                 LineWidth=2)
                    mcur.setPropertyValue('CharLeftBorder', border)
                    mcur.setPropertyValue('CharTopBorder', border)
                    mcur.setPropertyValue('CharRightBorder', border)
                    mcur.setPropertyValue('CharBottomBorder', border)

            if not cursor.gotoNextWord(False):
                break


# similar set of letters
vowels = list('aeiou')
c21 = list('aceo')
c22 = list('imnrsuvwxz')
c12 = list('bdhklt')
c23 = list('gjpqy')
subs = (vowels, c21, c22, c12, c23)
swaps = ('ie', 'ei')
end_sub = {
    'able': 'ible',
    'ible': 'able',
    'ance': 'ence',
    'ence': 'ance',
    'ceed': 'seed',
    'cede': 'sede',
    'ery': 'ary',
    'ary': 'ery',
    'ent': 'ant',
    'ant': 'ent',
    'eed': 'ede',
    'lly': 'ly',
    'eur': 'er',
    'al': 'el',
    'el': 'al',
    'te': 't',
    'mn': 'm',
    'll': 'l',
    'l': 'll'
}
mid_sub = {
    'sc': 'ch',
    'te': 'ght',
    'ght': 'te',
    'ate': 'eat',
    'eat': 'ate',
    'ten': 'tain',
    'nun': 'noun',
    'tain': 'ten'
}
full_sub = {
    'equipment': ['equiptment'],
    'accommodate': ['acommodate', 'accomodate'],
    'acknowledgment': ['acknowledgement'],
    'acquire': ['aquire'],
    'apparent': ['apparant', 'aparent', 'apparrent', 'aparrent'],
    'calendar': ['calender'],
    'colleague': ['collaegue', 'collegue', 'coleague'],
    'conscientious': ['consciencious'],
    'consensus': ['concensus'],
    'entrepreneur': ['entrepeneur', 'entreprenur', 'entreperneur'],
    'fulfill': ['fulfil'],
    'indispensable': ['indispensible'],
    'led': ['lead'],
    'laid': ['layed'],
    'liaison': ['liasion'],
    'license': ['licence', 'lisence'],
    'maintenance': ['maintainance', 'maintnance'],
    'necessary': ['neccessary', 'necessery'],
    'occasion': ['occassion'],
    'occurred': ['occured'],
    'pastime': ['pasttime'],
    'privilege': ['privelege', 'priviledge'],
    'publicly': ['publically'],
    'receive': ['recieve'],
    'recommend': ['recomend', 'reccommend'],
    'referred': ['refered'],
    'relevant': ['relevent', 'revelant'],
    'separate': ['seperate'],
    'successful': ['succesful', 'successfull', 'sucessful'],
    'underrate': ['underate'],
    'until': ['untill'],
    'withhold': ['withold']
}


def wrong(word: str):
    chars = list(word)
    l = len(chars)  # used to optionally remove few characters from the end

    # double to single
    i = 1
    while i < l:
        if chars[i] == chars[i-1] and random.random() < 3/len(chars):
            chars.pop(i)
            l -= 1
        else:
            i += 1

    # swap characters
    for i in range(1, len(chars), 2):
        c1, c2 = chars[i-1], chars[i]
        for o1, o2 in swaps:
            if c1 == o1 and c2 == o2 and random.random() < 3/len(chars):
                chars[i-1], chars[i] = chars[i], chars[i-1]

    # real english mistakes
    word = ''.join(chars)
    # full replace
    if word in full_sub and random.random() < 0.75:
        return random.choice(full_sub[word])
    # replace end
    for k, v in end_sub.items():
        if word.endswith(k):
            return word[:-len(k)] + v
    # replace middle
    for k, v in mid_sub.items():
        try:
            idx = word.index(k)
            return word[:idx] + v + word[idx+len(k):]
        except ValueError:
            pass

    # random mistakes
    t = random.randint(0, 4)
    if t == 0:  # replace characters
        for replacor in subs:
            for i in range(len(chars)):
                if chars[i].lower() in replacor and random.random() < .33:
                    nc = random.choice(replacor)
                    chars[i] = nc if chars[i].islower() else nc.upper()
                    if random.random() < .5:
                        l = i
                    break
    elif t == 1:  # swap characters
        for i in range(1, len(chars)-1):
            if random.random() < 2/len(chars):
                chars[i], chars[i+1] = chars[i+1], chars[i]
                if random.random() < .5:
                    l = i+1
    return ''.join(chars[:l])


def fix_upper(cursor):
    """Change font for upper case letters which are not followed by lower case letter."""
    l = len(cursor.String)
    for i in range(l):
        if (i == l-1 or not cursor.String[i+1].islower()) and cursor.String[i].isupper():
            cur = cursor.Text.createTextCursorByRange(cursor)
            cur.goLeft(l-i, False)
            cur.goRight(1, True)
            cur.CharFontName = single_char_font
