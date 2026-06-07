"""
Number parsing copied WHOLESALE from worldcode.py (numberParser / parseTextIntoNumbers /
parseNumbersOrRange / WeightedRange / ...). Only the GENERATION changed: instead of the
in-process token-trie + greedy/topN logit decode, worldRefactored.gen_number generates the
number TEXT with NUMBER_GRAMMAR and feeds it to these parsers. The handful of unconditional
debug prints in the originals were gated behind `debug`.

Needs:  pip install number_parser
"""
import re
import random
import traceback

import number_parser


def splitNoEmpty(text, splitStr=None):
    return [x.strip() for x in text.split(splitStr) if len(x.strip()) > 0]


def getRangePieces(number):
    if not '-' in number:
        return [number]
    else:
        pieces = [x.strip() for x in number.split("-") if len(x.strip()) > 0]
        return pieces


def parseDecimalIntoPieces(decimalPieces):
    if not '.' in decimalPieces:
        return decimalPieces.strip(), "0"
    else:
        pieces = [x.strip() for x in decimalPieces.split(".") if len(x.strip()) > 0]
        afterPeriod = '0'
        if len(pieces) > 1:
            afterPeriod = pieces[1]  # ignore stuff after an additional period (probably a phone number)
        beforePeriod = pieces[0]
        return beforePeriod, afterPeriod


MAGNITUDE_WORDS = ['trillion', 'billion', 'million', 'thousand', 'hundred']
TENS_WORDS = ['ninety', 'eighty', 'seventy', 'sixty', 'fifty', 'forty', 'thirty', 'twenty']
TEENS_WORDS = ['nineteen', 'eighteen', 'seventeen', 'sixteen', 'fifteen', 'fourteen', 'thirteen', 'twelve', 'eleven']
ONES_WORDS = ['nine', 'eight', 'seven', 'six', 'five', 'four', 'three', 'two', 'one']
NUMBER_WORDS = MAGNITUDE_WORDS + TENS_WORDS + TEENS_WORDS + ['ten', 'zero'] + ONES_WORDS


def splitCompoundNumbers(numberText, debug=False):
    # number_parser can't handle stuff like sixhundred; this splits them apart
    dashPieces = splitNoEmpty(numberText, "-")

    def splitApartNumberWord(word):
        splitPieces = []
        endI = len(word)
        # due to four and fourty overlapping we need to go back to front
        for startI in range(len(word) - 1, 0, -1):
            piece = word[startI:endI].lower()
            if piece in NUMBER_WORDS:
                splitPieces.append(piece)
                endI = startI
        remaining = word[:endI].strip()
        if len(remaining) > 0:
            splitPieces.append(remaining)
        splitPieces = splitPieces[::-1]  # reverse since we started from back
        return " ".join(splitPieces)

    def splitApartNumberWords(text):
        return " ".join([splitApartNumberWord(word) for word in splitNoEmpty(text)])

    splitApart = "-".join([splitApartNumberWords(text) for text in dashPieces])
    if debug:
        print("before split apart\n" + numberText + "\nafter split apart\n" + splitApart)
    return splitApart


def numberParser(numberText):
    return number_parser.parse(numberText).replace(",", "")


def charPositions(char, val):
    return [i for i in range(len(val)) if val[i] == char]


def splitIntoBeforeAndAfterDecimal(number):
    pieces = splitNoEmpty(number, ".")
    if len(pieces) > 2:
        pass  # warning: more than one period
    numZerosAfterPeriod = 0
    afterPeriod = 0
    if len(pieces) > 1:
        afterPeriod = "".join(pieces[1:])
        zerosAfterPeriodSpan = re.match("^0*", afterPeriod).span()
        numZerosAfterPeriod = zerosAfterPeriodSpan[1] - zerosAfterPeriodSpan[0]
        afterPeriod = int(afterPeriod)
    return int(pieces[0]), numZerosAfterPeriod, afterPeriod


class WeightedRange(object):
    def __init__(self, minVal, minPr, maxVal, maxPr):
        prSum = minPr + maxPr
        if prSum == 0:
            prSum = 1
        minPr = minPr / prSum
        maxPr = maxPr / prSum
        if maxVal < minVal:  # makes debug easier to read
            minVal, minPr, maxVal, maxPr = maxVal, maxPr, minVal, minPr
        self.minVal = minVal
        self.minPr = minPr
        self.maxVal = maxVal
        self.maxPr = maxPr

    def sample(self):
        # uniform weighting
        if self.minPr == self.maxPr:
            return random.random() * (self.maxVal - self.minVal) + self.minVal
        if self.maxVal == self.minVal:
            return self.maxVal
        if self.minPr == 0:
            return self.maxVal
        if self.maxPr == 0:
            return self.minVal

        def sampleRecursive(minVal, minPr, maxVal, maxPr, p, curIter, maxIters):
            prSum = minPr + maxPr
            if prSum == 0:
                prSum = 1
            minPr = minPr / prSum
            maxPr = maxPr / prSum
            middleVal = minVal * minPr + maxVal * maxPr
            middlePr = (minPr + maxPr) / 2.0
            nextMin, nextMinPr, nextMax, nextMaxPr = minVal, minPr, maxVal, maxPr
            if p < 0.5:
                nextMax = middleVal
                nextMaxPr = middlePr
                p = p * 2
            elif p > 0.5:
                nextMin = middleVal
                nextMinPr = middlePr
                p = (p - 0.5) * 2
            else:
                return middleVal
            if curIter >= maxIters:
                return nextMin * (1 - p) + nextMax * p
            else:
                return sampleRecursive(nextMin, nextMinPr, nextMax, nextMaxPr, p, curIter + 1, maxIters)

        p = random.random()
        return sampleRecursive(self.minVal, self.minPr, self.maxVal, self.maxPr, p, 0, 15)

    def __repr__(self):
        if self.minPr == self.maxPr:
            return f"[{self.minVal}, {self.maxVal}]"
        else:
            return f"[{self.minVal}:{self.minPr}, {self.maxVal}:{self.maxPr}]"


def parseIntRangeFromText(numberText, debug=False):
    num = parseNumbersOrRange(numberText, debug=debug)
    minVal = num['pieces'][0][0]   # 1 is numZerosAfterPeriod, 2 is num afterPeriod
    maxVal = num['pieces'][-1][0]
    return WeightedRange(minVal=minVal, minPr=0.5, maxVal=maxVal, maxPr=0.5)


def parseNumbersOrRange(numberText, debug=False):
    try:
        onlyNumbers = parseTextIntoNumbers(numberText, debug=debug)
        rangePieces = [splitIntoBeforeAndAfterDecimal(x) for x in getRangePieces(onlyNumbers)]
        if len(rangePieces) == 1:
            return {"type": "value", "pieces": rangePieces}
        else:
            return {"type": "range", "pieces": rangePieces}
    except Exception:
        if debug:
            print('failed to parse to numbers in parseNumbersOrRange("' + numberText + '")')
            print(traceback.format_exc())
        return {"type": "error", "pieces": []}


# turns stuff like 3.5 1000 into 3500
def expandDecimalThanOneAndZeros(text):
    pieces = splitNoEmpty(text)
    parsed = [False for _ in range(len(pieces))]
    parsedPieces = []
    for i in range(len(pieces)):
        if parsed[i]:
            continue
        if i == len(pieces) - 1:
            parsedPieces.append(pieces[i])
            continue
        curPiece = pieces[i]
        nextPiece = pieces[i + 1]
        nextWithoutCommas = nextPiece.replace(",", "")
        nextIsOneWithZeros = re.match("^10+$", nextWithoutCommas) is not None
        curIsNumber = re.match(r"^[0-9\.]+$", curPiece) is not None
        curIsRange = re.match(r"^[0-9\.]+\-[0-9\.]+$", curPiece) is not None
        if (curIsNumber or curIsRange) and nextIsOneWithZeros:  # e.g. 6.4 million -> 6400000
            rangePieces = getRangePieces(curPiece)
            outPieces = []
            for rangePiece in rangePieces:
                beforeDecimal, afterDecimal = parseDecimalIntoPieces(rangePiece)
                numZeros = nextPiece.count("0")
                if len(afterDecimal) < numZeros:
                    remainingZeros = numZeros - len(afterDecimal)
                    resNumber = beforeDecimal + afterDecimal + ("0" * remainingZeros)
                else:
                    stuffBeforeDecimal = afterDecimal[:numZeros]
                    stuffAfterDecimal = afterDecimal[numZeros:]
                    resNumber = beforeDecimal + stuffBeforeDecimal + "." + stuffAfterDecimal
                outPieces.append(resNumber)
            parsedPieces.append("-".join(outPieces))
            parsed[i + 1] = True
        else:
            parsedPieces.append(curPiece)
    return " ".join(parsedPieces)


def parseTextIntoNumbers(numberText, debug=False):
    # remove trailing periods (they don't matter and can confuse the below if after a word)
    numberText = re.sub(r"\.$", "", numberText)
    # split apart stuff like fortyfour because number_parser can't handle those
    numberText = splitCompoundNumbers(numberText, debug=debug)
    numberText = numberText.replace(",", "")
    # the only case this doesn't handle is stuff like 6.4 million, handle that manually
    res = expandDecimalThanOneAndZeros(numberParser(numberText))
    hasMultipleNumbers = len(splitNoEmpty(res)) > 1

    # turns "five", "six thousand" into "five thousand", "six thousand"
    def applyBonusTermsForLHS(lhs, rhs):
        lhs = lhs.replace("-", " ")
        rhs = rhs.replace("-", " ")
        lhsPieces = splitNoEmpty(lhs)
        rhsPieces = splitNoEmpty(rhs)
        lastNonSelectedTerm = 1
        for i, term in enumerate(rhsPieces):
            if not term.lower() in MAGNITUDE_WORDS:
                lastNonSelectedTerm = i
        selectedPieces = [x for x in rhsPieces[lastNonSelectedTerm + 1:]
                          if x.lower() in MAGNITUDE_WORDS and not x in lhsPieces[1:]]
        if len(selectedPieces) > 0 and debug:
            print("adding", selectedPieces, "to lhs of", lhsPieces)
        lhsPieces = lhsPieces + selectedPieces
        return " ".join(lhsPieces), " ".join(rhsPieces)

    hasInvalidDash = False
    # only allow dashes that make sense, like forty-four, not four-five or hundred-thousand
    if '-' in numberText:
        positionsOfDashes = charPositions('-', numberText)
        for dashPos in positionsOfDashes:
            beforeText = numberText[:dashPos].replace("-", " ")
            afterText = numberText[dashPos + 1:].replace("-", " ")
            beforeWords = splitNoEmpty(beforeText)
            afterWords = splitNoEmpty(afterText)
            if len(beforeWords) >= 1 and len(afterWords) >= 1:
                beforeWord = beforeWords[-1].lower()
                afterWord = afterWords[-1].lower()
                if beforeWord in TENS_WORDS and afterWord in ONES_WORDS:
                    pass
                else:
                    hasInvalidDash = True

    if debug:
        print("after first parse", res)
    if '-' in res or hasMultipleNumbers or hasInvalidDash:
        # figure out which '-' is the range separator (pick the one giving the smallest interval)
        positionsOfDashes = charPositions('-', numberText)
        bestlhs, bestrhs = None, None
        smallestIntervalSize = None
        for dashPos in positionsOfDashes:
            beforeText = numberText[:dashPos].replace("-", " ")
            afterText = numberText[dashPos + 1:].replace("-", " ")
            lhs, rhs = applyBonusTermsForLHS(beforeText, afterText)
            lhs = expandDecimalThanOneAndZeros(numberParser(lhs))
            rhs = expandDecimalThanOneAndZeros(numberParser(rhs))
            lhsParsed = splitNoEmpty(lhs)
            rhsParsed = splitNoEmpty(rhs)
            if len(lhsParsed) == 1 and len(rhsParsed) == 1:
                intervalSize = int(rhsParsed[0]) - int(lhsParsed[0])
                if intervalSize > 0:
                    if smallestIntervalSize is None or intervalSize < smallestIntervalSize:
                        smallestIntervalSize = intervalSize
                        bestlhs, bestrhs = lhs, rhs
        if bestlhs is None:
            if debug:
                print("Could not find a dash that is used for range in " + numberText)
            raise Exception("Could not find a dash that is used for range in " + numberText)
        res = bestlhs + " - " + bestrhs
    return res.strip()


# ---------------------------------------------------------------------------
# Generation: the grammar that replaces parseNumber's token-trie + greedy/topN decode.
# It permits number WORDS (so the parser above does the heavy lifting), digits, magnitudes,
# and an optional range. Ends in "\n"; callers pass stop=["\n"] (terminator-safe).
# ---------------------------------------------------------------------------
def _alt(words):
    return " | ".join('"%s"' % w for w in words)


# A phrase must START with a value (digit or number-word), so the grammar can't emit a bare
# magnitude like "million" (which won't parse); mags / "and" may follow. Ranges use "-"
# (parser's getRangePieces splits on "-").
NUMBER_GRAMMAR = (
    'root ::= " " expr "\\n"\n'
    'expr ::= phrase (" - " phrase)?\n'
    'phrase ::= value (" " (value | mag | conn))*\n'
    'value ::= digits | numword\n'
    'digits ::= [0-9]+ ("." [0-9]+)?\n'
    'numword ::= ' + _alt(ONES_WORDS + TENS_WORDS + TEENS_WORDS + ['ten', 'zero']) + '\n'
    'mag ::= ' + _alt(MAGNITUDE_WORDS) + '\n'
    'conn ::= "and"'
)
