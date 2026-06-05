# how to install llama_cpp https://github.com/abetlen/llama-cpp-python/issues/400
'''
set FORCE_CMAKE=1
set "CMAKE_ARGS=-DLLAMA_CUBLAS=on"
set "CUDAFLAGS=-arch=all -lcublas"
pip3 install llama-cpp-python==v0.1.78 --force-reinstall --upgrade --no-cache-dir
that version is needed to support ggml, gguf was giving me worse results
'''

#import torch # important to do this first to prevent circular import issues

#from ctransformers import AutoModelForCausalLM
from llama_cpp import Llama
import llama_cpp
import number_parser
import traceback
import datetime
import codecs
import jsonpickle
import itertools
from collections import defaultdict
import heapq


from sortedcontainers import SortedList
from importlib import reload


import random
import time
import re
from datetime import timedelta

import numpy as np

import math

#from transformers import AutoTokenizer

def timeInMillis():
    return time.time_ns() // 1_000_000

MAX_TOKENS = 2048
N_THREADS = 8


NEGATIVE_LARGE = -9999999999999.0

specialSquare = '░'


# ideas:
# display a character's schedule like we display calenders

def example(model):
    print("generating character")
    eugene = Character(model=model, name='Eugene', worldDescription='Modern day Earth', species='giant isopod', gender='female')
    print("generating locations")
    locs = getLocations2(model=model, characterPrefix=eugene.characterPrefix, gender=eugene.gender, name=eugene.name, numLocations=20)
    print("getting pr char is in those locations")
    prs = getPrLocations(model=model, character=eugene, locations=locs['locations'])
    displayPrLocations(prs)
    for locType, locPrs in prs:
        print("getting names for", locType)
        
        locNames = getLocationName(model=model, char=eugene, locationType=locType, numNames=10, debug=False)
        # todo: if they live there, allow their name in the description, else probably don't
        #  exceptions like a shrine of a certain god, a local company named after the owner, etc.
        locNames = [l for l in locNames if not eugene.name.lower() in l.lower()]
        # todo: are there multiple of this type of location that char visits? Or just one?
        # todo: is char always in this type of location? (like earth), then it might be too broad
        # todo: add some repetition penalty to schedules so they don't get stuck in repeated phrases like "he loved to" etc.
    # if a schedule says "in a" or "at a" or "inside a" etc. then that gives hints about where it happens, extract that
    # todo: one time a schedule just output " the peaceful atmosphere", why?
def testNumMatchLeft():
  assert(numMatchLeft(baseStr="abcdefg", lookupStr="abcdefg")==7)
  assert(numMatchLeft(baseStr="abcdef", lookupStr="abcdefg")==6)
  assert(numMatchLeft(baseStr="abcdefg", lookupStr="abcdef")==0)
  assert(numMatchLeft(baseStr="wwabcdefg", lookupStr="abcdefg")==7)
  assert(numMatchLeft(baseStr="abcdefg", lookupStr="wwabcdefg")==0)
  assert(numMatchLeft(baseStr="abcabcdabcabce", lookupStr="abc")==0)
  assert(numMatchLeft(baseStr="aabcabcdabcabce", lookupStr="aabcabc")==0)
  assert(numMatchLeft(baseStr="abcabcdabcabce", lookupStr="dabcabc")==0)
  assert(numMatchLeft(baseStr="dbcabcdabcabc", lookupStr="dabcabc")==7)
  assert(numMatchLeft(baseStr="dbcabcdabcab", lookupStr="dabcabc")==6)

def numMatchLeft(baseStr, lookupStr):
  amount = 0
  for i in range(len(lookupStr),0,-1):
    cutoff = lookupStr[:i]
    if baseStr.endswith(cutoff):
      amount = i
      break
  return amount


class Model(object):
    def __init__(self, *args, **kwargs):
        #if len(args) == 0 and len(kwargs) == 0:
        #    args = [r"nous-hermes-13b.ggmlv3.q4_1.bin"]
        #    kwargs = {'model_type': 'llama', 'gpu_layers': 30, 'context_length': 2048}
        #self.llm = AutoModelForCausalLM.from_pretrained(*args, **kwargs)
        #self.tokenizer = AutoTokenizer.from_pretrained("llamatokenizer")
        self.llm = Llama(model_path="Y:/ai/openhermes-2.5-mistral-7b.Q5_K_M.gguf", n_gpu_layers=64, n_ctx=2048)
        
        #self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.llm.verbose = False
        self.n_vocab = self.llm.n_vocab()
        self.context = []
        self.fullDebug = False
    
    def generatep(self, instruction, prompt, responsePrefix, *args, **kwargs):
        for r in self.generateLong(instruction, prompt, responsePrefix, *args, **kwargs):
            if not self.fullDebug:
              print(r, end='', flush=True)
        if not self.fullDebug: print("")
        
    def reset(self, clear_cache=True):
        self.context = []
        self.llm.reset()
        if clear_cache:
            self.llm._ctx.kv_cache_clear()
    
    # NOTE! THIS STRIPS THE BOS TOKEN BY DEFAULT
    def evals(self, s, includeBos=False, logits_all=False):
        toks = self.tokenize(s)
        if not includeBos:
            toks = toks[1:]
        return self.eval(tokens=toks, logits_all=logits_all)
    def eval(self, tokens, logits_all=False, reset=False, debug=False):
        # modified from llama_cpp_python
        
        
        if reset:
            matchingPrefixLen = 0
            for i in range(min(len(tokens), len(self.context))):
                if tokens[i] == self.context[i]:
                    matchingPrefixLen += 1
                else:
                    break
            if matchingPrefixLen == 0:
                self.reset()
            # just pop back, this saves some time if overlap in prompts
            else:
                numToPop = len(self.context)-matchingPrefixLen
                if debug: print("overlap in prompt, popping back", numToPop)
                if numToPop > 0:
                  if debug: print("popping prompt:")
                  if debug: print(repr(self.detokenize(self.context[-numToPop:])))
                for i in range(numToPop):
                    self.popcontext()
                if debug: print("overlapped prompt")
                if debug: print(repr(self.detokenize(self.context)))
                tokens_ = tokens[matchingPrefixLen:]
                if debug: print("remaining prompt")
                if debug: print(repr(self.detokenize(tokens_)))
                if len(tokens_) == 0: # full overlap, need to pop one more to get logits
                  if debug: print("full overlap, popping back one more for logits")
                  self.popcontext()
                  tokens_ = tokens[matchingPrefixLen-1:]
                  if debug: print("remaining prompt")
                  print(repr(self.detokenize(tokens_)))
                # otherwise its fine since we are going to eval more tokens
                
                tokens = tokens_
                
        
        batch_size = self.llm.context_params.n_batch
    
        if hasattr(self, "logits"):
            del self.logits

        self.logits = None
        if logits_all and batch_size < len(tokens):
            self.logits = np.zeros([len(tokens)*self.n_vocab])
        for b in range(0, len(tokens), batch_size):
            batch_end = min(len(tokens), b+batch_size)
            cur_tokens = tokens[b:batch_end]
            n_past = len(self.context)
            self.context += cur_tokens
            n_tokens = len(cur_tokens)
            self.llm._batch.set_batch(batch=cur_tokens, n_past=n_past, logits_all=logits_all)
            self.llm._ctx.decode(self.llm._batch)
            offset = (0 if logits_all else n_tokens-1)
            cols = self.n_vocab
            rows = n_tokens
            if batch_end != len(tokens) and not logits_all:
                pass
            elif batch_end != len(tokens) and logits_all:
                self.logits[b*cols:batch_end*cols] = self.llm._ctx.get_logits()[offset * cols : rows * cols]
            elif batch_end == len(tokens) and (not logits_all or len(tokens) <= batch_size):
                self.logits = self.llm._ctx.get_logits()[offset * cols : rows * cols]
            elif batch_end == len(tokens) and logits_all:
                self.logits[b*cols:batch_end*cols] = self.llm._ctx.get_logits()[offset * cols : rows * cols]            
         
        '''
        n_past = len(self.context)
        self.context += tokens
        print("context", self.context)
        print("n_past", n_past)
        #n_past = len(self.context) - 1
        llama_cpp.llama_eval(
            ctx=self.llm.ctx,
            tokens=(llama_cpp.llama_token * len(tokens))(*tokens),
            n_tokens=llama_cpp.c_int(len(tokens)),
            n_past=llama_cpp.c_int(n_past),
            #n_threads=llama_cpp.c_int(N_THREADS),
        )
        offset = len(tokens)-1
        print("n_vocab", self.llm.n_vocab())
        print("offset", offset)
        
        logits_view = llama_cpp.llama_get_logits(self.llm.ctx)
        logits = logits_view[offset*self.n_vocab:(offset+1)*self.n_vocab]
        self.logits = logits
        #self.llm.eval(tokens)
        #self.logits = np.array(
        '''
        
    def tokenize(self, val):
        # weird hack we have to do to make it not stick a space in front of everything
        # the trick is just stick a special symbol in front instead then cut it away afterwards
        if type(val) is str:
            val = specialSquare + val
            val = val.encode("utf-8", "ignore")
        return [self.llm.token_bos()] + self.llm.tokenize(val)[3:] # note this number might need to be changed depending on special square and tokenizer
    
    def detokenize(self, tokens):
        while len(tokens) > 0 and tokens[0] == self.llm.token_bos():
          tokens = tokens[1:]
        bytesOfToks = self.llm.detokenize(tokens)
        return bytesOfToks.decode("utf-8", "ignore")
    
    def popcontext(self):
        # -1 seq_id matches all sequences
        if len(self.context) > 0:
            p0 = len(self.context)-1
            p1 = len(self.context)
            self.llm._ctx.kv_cache_seq_rm(seq_id=-1, p0=p0, p1=p1) 
            self.context.pop()
    
    def idToToken(self, id):
        return self.tokenizer._tokenizer.id_to_token(id)
    
    def tokenToId(self, token):
        return self.tokenizer._tokenizer.token_to_id(token)
        
    def generateFromQ(self, prompt, responsePrefix="", debug=True, *args, **kwargs):
        res = []
        if debug:
            print(toPrompt(prompt, responsePrefix), end='', flush=True)
        for willContinue, r in self.generate(toPrompt(prompt, responsePrefix), debug=debug, *args, **kwargs):
            if not r is None:
                res.append(r)
                if debug: print(r, end='', flush=True)
        
        if debug: print("")
        res = "".join(res)
        return res   
     
    def generateLong(self, instruction, prompt, responsePrefix, debug=False, *args, **kwargs):
        fullPrompt = toPromptFancy(instruction, prompt, responsePrefix)
        promptLen = len(self.tokenize(fullPrompt))
        noResponsePrefixLen = len(self.tokenize(toPromptFancy(instruction, prompt, "")))
        print(fullPrompt, end="", flush=True)
        resultTokens = []
        while True:
            done = False
            for willContinue, t in self.generate(fullPrompt, debug, *args, **kwargs):
                if t is None:
                    done = not willContinue
                    break
                else:
                    resultTokens.append(t)
                    yield t
            if not done: # need to continue prompt
                fullPrompt = toPromptFancy(instruction, prompt, "".join(resultTokens[-MAX_TOKENS+MAX_TOKENS//3+noResponsePrefixLen:]))
                print("continuing with prompt")
                print(fullPrompt)
            else:
                break
        
        
    def sample(self,
        top_k=40,
        top_p=0.95,
        min_p=0.05,
        typical_p=1.0,
        temp=0.8,
        tfs_z=1.0,
        logits_processor=None,
        ):
        logits = self.logits
        
        # from llama-cpp-python
        if logits_processor is not None:
            logits[:] = logits_processor(self.context, logits)        
        
        self.llm._candidates.copy_logits(logits)
        #self.llm._ctx.sample_top_k(candidates=self.llm._candidates, k=top_k, min_keep=1)
        #self.llm._ctx.sample_tail_free(candidates=self.llm._candidates, z=tfs_z, min_keep=1)
        #self.llm._ctx.sample_typical(
        #    candidates=self.llm._candidates, p=typical_p, min_keep=1
        #)
        #self.llm._ctx.sample_top_p(candidates=self.llm._candidates, p=top_p, min_keep=1)
        #self.llm._ctx.sample_min_p(candidates=self.llm._candidates, p=min_p, min_keep=1)
        self.llm._ctx.sample_temp(candidates=self.llm._candidates, temp=temp)
        token = self.llm._ctx.sample_token(candidates=self.llm._candidates)
        return token

    def generateIter(self, *args, **kwargs):
        while True:
            t = self.sample(*args, **kwargs)
            yield t
            if t == self.llm._token_eos: break
            self.eval([t])
        
    def rollback(self, genResults, logits): # one less because we interrupt before eval
        for t in s['gentok'][:-1]:
            self.popcontext()
        self.logits = logits
    
    # strictStop means that stop strings are forbidden at the front (starting from forbidding the first token of them)
    # if strictStop is false, the first few tokens of a multi-token stopString can be genererated at the front, it's just the last tok that is forbidden 
    def generate2(self, stop=None, stopFunc=None, rollback=False, numTokens=-1, debug=False, strictStop=False, *args, **kwargs):
        generatedTokens = []
        outputTokens = []
        if rollback:
            backupLogits = self.logits
        if not stop is None:
            initialContextSize = len(self.context)
            old_logits_processor = kwargs['logits_processor'] if 'logits_processor' in kwargs else None
            stopTokens = [self.tokenize(s)[1:] for s in stop]
            longest = max([len(toks) for toks in stopTokens])
            # prevents generation from doing stop tokens immediately
            def stop_token_logit_processor(inds, logits):
                # forbid immediately outputting end of text token
                if len(inds) == initialContextSize:
                    logits[self.llm._token_eos] = NEGATIVE_LARGE
                if len(inds) <= initialContextSize+longest:
                    newContext = self.context[initialContextSize:]
                    stopToksBeingGenerated = []
                    for toks in stopTokens:
                        beingGenerated = True
                        for i, t in enumerate(toks[:len(newContext)]):
                            if newContext[i] != t:
                                beingGenerated = False
                                break
                        if beingGenerated and (strictStop or len(toks) == len(newContext)+1):
                            stopToksBeingGenerated.append(toks[len(newContext)])
                            if debug: print("being generated", toks, f'{repr(self.detokenize(toks))}', "so forbidden", stopToksBeingGenerated[-1], "which is", f'"{self.detokenize([stopToksBeingGenerated[-1]])}"')
                    for t in stopToksBeingGenerated:
                        if debug: print("stop tok forbidden", t, f'{repr(self.detokenize([t]))}')
                        logits[t] = NEGATIVE_LARGE
                        
                if not old_logits_processor is None:
                    logits = old_logits_processor(inds, logits)
                return logits
            kwargs['logits_processor'] = stop_token_logit_processor
        
        for t in self.generateIter(*args, **kwargs):
            generatedTokens.append(t)
            outputTokens.append(t)
            curStr = self.detokenize(outputTokens)
            if not stop is None:
                stopping = any([(s in curStr) for s in stop])
                if stopping: # don't include stop str
                    
                    
                    stopStrs = [s for s in stop if (s in curStr)]
                      # look how far we need to rollback until stop str is no longer in output at all
                    numRollback = 1
                    for s in stopStrs:
                      while True:
                        numPrefix = numMatchLeft(baseStr=self.detokenize(generatedTokens[:-numRollback]), lookupStr=s)
                        if numPrefix == 0:
                          break
                        numRollback += 1
                    
                    outputTokens = generatedTokens[:-numRollback]
                    # we are gonna rollback everything anyway, don't bother
                    if rollback or numRollback == 1: # or we only rollback one which isn't any
                      pass
                    else:
                      print("rolling back multiple", numRollback)
                      numToPopContext = numRollback - 1 # we don't eval current token
                      
                      for popi in numToPopContext:
                        self.popcontext()
                      
                      # rollback one more so we can make logits correct
                      curTok = self.context[-1]
                      self.popcontext()
                      self.eval(curTok)
                      
                    
                    break
            if numTokens >= 0:
                if len(outputTokens) >= numTokens:
                    break            
            if not stopFunc is None:
                stopFuncRes = stopFunc(s=curStr, toks=generatedTokens)
                if stopFuncRes['stop']:
                    if 'includeLast' in stopFuncRes and not stopFuncRes['includeLast']:
                        outputTokens = outputTokens[:-1]
                    break
        resStr = self.detokenize(outputTokens)
        if rollback:
            for t in generatedTokens[:-1]: # one less because we interrupt before eval
                self.popcontext()
            self.logits = backupLogits
        return {"str": resStr, "tok": outputTokens, "gentok": generatedTokens}
        
        
                    
        
        
    def generate(self, prompt, debug=False, *args, **kwargs):
        if self.fullDebug is True:
          print(prompt, end="", flush=True)
        firstResponse = True
        startTimeMillis = timeInMillis()
        startTimeAfterFirst = timeInMillis()
        numTokens = 0
        willContinue = True
        promptLen = len(self.tokenize(prompt))
        stopTokens = []
        if 'stop' in kwargs:
            stopTokens = kwargs['stop']
            del kwargs['stop']
            if debug: print("stop of", str(stopTokens))
        maxTokens = MAX_TOKENS
        customMaxTokens = False
        if 'max_tokens' in kwargs:
            maxTokens = kwargs['max_tokens']
            del kwargs['max_tokens']
            customMaxTokens = True
        eosToken = self.llm._token_eos
        self.eval(self.tokenize(prompt))
        tokenIter = self.generateIter(*args, **kwargs)
        toksGenerated = []
        strGeneratedSoFar = ""
        while True:
            nextToken = next(tokenIter, "finished")
            if nextToken == "finished" or nextToken == eosToken:
                willContinue = False
                break
            numTokens += 1
            if firstResponse:
                startTime = timeInMillis() - startTimeMillis
                firstResponse = False
                startTimeAfterFirst = timeInMillis()
            tok = self.detokenize([nextToken])
            if self.fullDebug is True:
              print(tok, end="", flush=True)
            yield True, tok
            foundStopToken = False
            if type(stopTokens) is list:
                for stopt in stopTokens:
                    if stopt in tok:
                        foundStopToken = True
            else:
                foundStopToken = stopTokens(strGeneratedSoFar, tok, toksGenerated, nextToken)
            toksGenerated.append(nextToken)
            strGeneratedSoFar += tok
            if numTokens+promptLen >= maxTokens:
                willContinue = not customMaxTokens
                break
            if foundStopToken:
                willContinue = False
                break
            
        if debug: print("\nstart millis", startTime)
        if debug: print("num tokens", numTokens, "millis per token", (timeInMillis() - startTimeAfterFirst) / max(1, numTokens-1))
        yield willContinue, None
        
    def __del__(self):
        del self.llm
        
def tests():
    numberTest()

def testEq(a, b):
    if a != b:
        print("failed", a, "!=", b)
    assert(a == b)

def numberTest():
    testEq(parseTextIntoNumbers("fifty four"), "54")
    testEq(parseTextIntoNumbers("6.4 million"), "6400000")
    testEq(parseTextIntoNumbers("3.46-5.6 thousand"), "3460 - 5600")
    testEq(parseTextIntoNumbers("fifty-four-six hundred and seventynine hundred thousand"), "5400000 - 67900000")
    testEq(parseTextIntoNumbers("thirty-three-million-fifty-four-million"), "33000000 - 54000000")
    testEq(parseTextIntoNumbers("sixty four hundred-sixty five hundred thousand"), "6400000 - 6500000") # TODO
    #testEq(parseTextIntoNumbers("twenty five - fifty three thousand hundred"), "todo") # TODO
    #testEq(parseTextIntoNumbers("fifty four million-thirty three million"), 'todo') # TODO
    testEq(parseTextIntoNumbers("thirty three million-fifty four million"), '33000000 - 54000000') # TODO
    testEq(parseTextIntoNumbers(" 10 million."), '10000000')
    print("passed parseTextIntoNumbers()")

def parseDecimalIntoPieces(decimalPieces):
    if not '.' in decimalPieces:
        return decimalPieces.strip(), "0"
    else:
        pieces = [x.strip() for x in decimalPieces.split(".") if len(x.strip()) > 0]
        afterPeriod = '0'
        if len(pieces) > 1:
            afterPeriod = pieces[1] # ignore stuff after an additional period, it's probably a phone number or something
        beforePeriod = pieces[0]
        return beforePeriod, afterPeriod

def getRangePieces(number):
    if not '-' in number:
        return [number]
    else:
        pieces = [x.strip() for x in number.split("-") if len(x.strip()) > 0]
        return pieces
        
def splitNoEmpty(text, splitStr=None):
    return [x.strip() for x in text.split(splitStr) if len(x.strip()) > 0]
    
def splitCompoundNumbers(numberText, debug=False):
     # number_parser can't handle stuff like sixhundred, this splits them apart
    dashPieces = splitNoEmpty(numberText, "-")
    def splitApartNumberWord(word):
        splitPieces = []
        endI = len(word)
        # due to four and fourty overlapping we need to go back to front
        for startI in range(len(word)-1, 0, -1):
            piece = word[startI:endI].lower()
            #print(piece)
            if piece in NUMBER_WORDS:
                splitPieces.append(piece)
                #print("got word", piece, "from", word)
                endI = startI
        remaining = word[:endI].strip()
        if len(remaining) > 0:
            splitPieces.append(remaining)
        splitPieces = splitPieces[::-1] # reverse since we started from back
        return " ".join(splitPieces)
    def splitApartNumberWords(text):
        return " ".join([splitApartNumberWord(word) for word in splitNoEmpty(text)])
    splitApart = "-".join([splitApartNumberWords(text) for text in dashPieces])
    if debug:
        print("before split apart")
        print(numberText)
        print("after split apart")
        print(splitApart)
    return splitApart
    
def numberParser(numberText):
    return number_parser.parse(numberText).replace(",", "")
        
def charPositions(char, val):
    return [i for i in range(len(val)) if val[i] == char]
    

MAGNITUDE_WORDS = ['trillion', 'billion', 'million', 'thousand', 'hundred']
TENS_WORDS = ['ninety', 'eighty', 'seventy', 'sixty', 'fifty', 'forty', 'thirty', 'twenty']
TEENS_WORDS = ['nineteen', 'eighteen', 'seventeen', 'sixteen', 'fifteen', 'fourteen', 'thirteen', 'twelve', 'eleven']
ONES_WORDS = ['nine', 'eight', 'seven', 'six', 'five', 'four', 'three', 'two', 'one']   

NUMBER_WORDS = MAGNITUDE_WORDS + TENS_WORDS + TEENS_WORDS + ['ten', 'zero'] + ONES_WORDS

def splitIntoBeforeAndAfterDecimal(number):
    pieces = splitNoEmpty(number, ".")
    if len(pieces) > 2:
        print("warning: more than one period", number)
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
    # normalize prs
    prSum = minPr + maxPr
    if prSum == 0: prSum = 1
    minPr = minPr/prSum
    maxPr = maxPr/prSum
    
    # technically the logic works fine if they are swapped but this makes debug easier to read
    if maxVal < minVal:
      minVal, minPr, maxVal, maxPr = maxVal, maxPr, minVal, minPr
    
    self.minVal = minVal
    self.minPr = minPr
    self.maxVal = maxVal
    self.maxPr = maxPr
  
  def sample(self):
    print("\nsampling from", str(self))
    # uniform weighting
    if self.minPr == self.maxPr:
      return random.random()*(self.maxVal-self.minVal)+self.minVal
    
    # no need for sample if same
    if self.maxVal == self.minVal:
      return self.maxVal
    
    # no need for sample if one of the prs is 0
    if self.minPr == 0: return self.maxVal
    if self.maxPr == 0: return self.minVal
      
    # if they are both 0.5, I want to do uniform min to max
    # if min is 0 and max is 1, I want to do only max
    # say min is 0.6 and max is 0.4
    # ok one way to think about it is that if both 0.5, then every number between and including endpoints has equal weighted
    # now if I decrease max, some of that end weight is moved to front
    # we can represent this by 
    # hmm if we do a weighted random sample of min and max, that'll give us which one to pick
    # another way to think of it is to convert min and max weight into values greater than one (respecting weight ratio)
    # for example, 0.2 and 0.8 is 1/5 and 4/5, which can turn into 1 and 4
    # then min gets weight 1, max gets weight 4
    # everything inbetween smoothly interpolates between these weights
    # that should work as long as they aren't order of magnitude difference?
    # ok then we can renormalize and it's just like smoothly interpolating original weights
    # if we have n with weight p and m with weight q,
    # and every number k inbetween has weight computed by (1-d)*p+d*q, where d=(k-n)/(m-n) (d is 0 at n and 1 at m)
    # how do we randomly sample from that?
    # normally to randomly sample from weighted distr we get a 0-1 then keep cumulating sum until it's bigger than us
    # though note that this will be bigger than 1, what is total sum?
    # eww
    # maybe easier way, we add stuff on end to compensate for weight
    # no that doesn't really work
    # basically we want to "pull" towards the side that has higher weight
    # what if we just compute where 0.5 is?
    # like by default it's in middle
    # but instead 0.5 is just at n*p+m*q
    # now we can view it as
    # n has weight p and middle has weight (p+q)/2 (then we normalize)
    # ok that gives us an algorithm that lets us binary search:
    # 1. Generate random number from 0 to 1
    # 2. Find 0.5 location, then split and recurse until we are close enough
    def sampleRecursive(minVal, minPr, maxVal, maxPr, p, curIter, maxIters):
      # 1: normalize endpoint prs
      prSum = minPr + maxPr
      if prSum == 0: prSum = 1
      minPr = minPr/prSum
      maxPr = maxPr/prSum
      
      # 2: find location of 0.5
      # it's a weighted sum of min and max
      # with pr the average of pr of min and max
      middleVal = minVal*minPr + maxVal*maxPr
      middlePr = (minPr + maxPr)/2.0
      #print("minVal", minVal, "minPr", minPr, "p", p, "maxVal", maxVal, "maxPr", maxPr)
      #print("curIter", curIter, "maxIters", maxIters)
      nextMin = minVal
      nextMinPr = minPr
      nextMax = maxVal
      nextMaxPr = maxPr
      
      if p < 0.5:
        nextMax = middleVal
        nextMaxPr = middlePr
        # scale 0-0.5 to 0-1
        p = p*2
      elif p > 0.5:
        nextMin = middleVal
        nextMinPr = middlePr
        # shift 0.5-1.0 to 0-0.5
        p = (p-0.5)
        # scale 0-0.5 to 0-1
        p = p*2
      else:
        return middleVal
      # done binary search, just do weighted avg of min and max
      if curIter >= maxIters:
        return nextMin*(1-p) + nextMax*p
      # recurse binary searchh
      else:
        return sampleRecursive(minVal=nextMin, minPr=nextMinPr, maxVal=nextMax, maxPr=nextMaxPr, p=p, curIter=curIter+1, maxIters=maxIters)
    
    p = random.random()
    return sampleRecursive(minVal=self.minVal, minPr=self.minPr, maxVal=self.maxVal, maxPr=self.maxPr, p=p, curIter=0, maxIters=15)
  
  def __repr__(self):
    if self.minPr == self.maxPr:
      return f"[{self.minVal}, {self.maxVal}]"
    else:
      return f"[{self.minVal}:{self.minPr}, {self.maxVal}:{self.maxPr}]"
  

def testExemptStrings(model):
    model.reset()
    prompt = "Testing hello how"
    def gen(prompt, n_tokens, *args, **kwargs):
        toks = []
        for willContinue, t in model.generate(prompt, *args, **kwargs):
            
            toks.append(t)
            if len(toks) >= n_tokens: break
            if not willContinue: break
        return "".join(toks)
    n_context = len(model.tokenize(prompt))

    results = []
    numRuns = 100
    for i in range(numRuns):
        model.reset()
        exemptProcessor = exemptStringsProcessor(model, n_context=n_context, forbiddenStrings=results)
        res = gen(prompt=prompt, n_tokens=5, logits_processor=exemptProcessor)
        print(prompt + res)
        results.append(res)
    if len(set(results)) != numRuns:
        print("Error: some duplicated element")
    else:
        print("Success: no duplicated outputs")
        
        
def exemptStringsProcessor(model, n_context, forbiddenStrings, mode='excludeFirst', debug=False):
    # simplest thing to do is just prevent first token match
    # second best thing is exclude non-filler words
    forbiddenToks = [model.tokenize(s)[1:] for s in forbiddenStrings]
    forbiddenToks = [toks for toks in forbiddenToks if len(toks) > 0] # exclude empty
    
    if mode == 'excludeFirst':
        
        def logits_processor(logits, index):
            if index == 0:
                if debug:
                    print("before:")
                    maxinds = np.argsort(-logits)
                    for t in maxinds[:20]:
                        print(t, model.detokenize([t]), logits[t])
                    
                    print("   ")
                
                if debug: print("removing")
                for toks in forbiddenToks:
                    if debug: print(model.detokenize(toks))
                    if debug: print(toks[0], model.detokenize([toks[0]]), logits[toks[0]])
                    logits[toks[0]] = NEGATIVE_LARGE
                    if debug: print(toks[0], model.detokenize([toks[0]]), logits[toks[0]])
                if debug: print()
                if debug:
                    print("after:")
                    print("   ")
                    maxinds = np.argsort(-logits)
                    for t in maxinds[:20]:
                        print(t, model.detokenize([t]), logits[t])
                
                    print("   ffffff")
            return logits
    
    def final_logits_processor(input_ids, logits):
        index = len(input_ids) - n_context
        if debug: print()
        if debug: print("..." + model.detokenize(input_ids[-10:]), index)
        if debug: print()
        res = logits_processor(logits=logits, index=index)
        return res
    return final_logits_processor
    
        

  
def parseIntRangeFromText(numberText, debug=False):
  print("got number text", numberText)
  num = parseNumbersOrRange(numberText)
  print("got output", num)
  minVal = num['pieces'][0][0] # 1 is numZerosAfterPeriod, 2 is num afterPeriod
  maxVal = num['pieces'][-1][0] # 1 is numZerosAfterPeriod, 2 is num afterPeriod
  print('minVal', minVal, 'maxVal', maxVal)
  return WeightedRange(minVal=minVal, minPr=0.5, maxVal=maxVal, maxPr=0.5)

def parseNumbersOrRange(numberText, debug=False):
    try:
        onlyNumbers = parseTextIntoNumbers(numberText, debug=debug)
        rangePieces = [splitIntoBeforeAndAfterDecimal(x) for x in getRangePieces(onlyNumbers)]
        if len(rangePieces) == 1:
            return {"type": "value", "pieces": rangePieces}
        else:
            return {"type": "range", "pieces": rangePieces}
    except:
        print('failed to parse to numbers in parseNumbersOrRange("' + numberText + '")')
        print(traceback.format_exc())
        return {"type": "error", "pieces": []}
      
        
# turns stuff like 3.5 1000 into 3500
def expandDecimalThanOneAndZeros(text):
    pieces = splitNoEmpty(text)
    parsed = [False for _ in range(len(pieces))]
    parsedPieces = []
    for i in range(len(pieces)):
        if parsed[i]: continue
        if i == len(pieces)-1:
            parsedPieces.append(pieces[i]) 
            continue
        
        curPiece = pieces[i]
        nextPiece = pieces[i+1]
        nextWithoutCommas = nextPiece.replace(",", "")
        nextIsOneWithZeros = re.match("^10+$", nextWithoutCommas) != None
        curIsNumber = re.match(r"^[0-9\.]+$", curPiece) != None
        curIsRange = re.match(r"^[0-9\.]+\-[0-9\.]+$", curPiece) != None
        # something like 6.4 million, turn into 6400000
        if (curIsNumber or curIsRange) and nextIsOneWithZeros:
            rangePieces = getRangePieces(curPiece)
            outPieces = []
            for rangePiece in rangePieces:
                beforeDecimal, afterDecimal = parseDecimalIntoPieces(rangePiece)
                numZeros = nextPiece.count("0")
                if len(afterDecimal) < numZeros:
                    remainingZeros = numZeros - len(afterDecimal)
                    resNumber = beforeDecimal + afterDecimal + ("0"*remainingZeros)
                else:
                    # too many decimal specified, our number is still something.something
                    stuffBeforeDecimal = afterDecimal[:numZeros]
                    stuffAfterDecimal = afterDecimal[numZeros:]
                    resNumber = beforeDecimal + stuffBeforeDecimal + "." + stuffAfterDecimal
                outPieces.append(resNumber)
            parsedPieces.append("-".join(outPieces))
            parsed[i+1] = True
        else:
            parsedPieces.append(curPiece)
    return " ".join(parsedPieces)
        
def parseTextIntoNumbers(numberText, debug=False):
    
    # remove trailing periods (they don't matter and can confuse the below if they are after a word)
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
        for i, term in enumerate(rhsPieces): # addresses stuff like fifty-four-six hundred and seventynine hundred thousand
            if not term.lower() in MAGNITUDE_WORDS:
                lastNonSelectedTerm = i
        
        selectedPieces = [x for x in rhsPieces[lastNonSelectedTerm+1:] if x.lower() in MAGNITUDE_WORDS and not x in lhsPieces[1:]]
        if len(selectedPieces) > 0 and debug:
            print("adding", selectedPieces, "to lhs of", lhsPieces)
        lhsPieces = lhsPieces + selectedPieces
        return " ".join(lhsPieces), " ".join(rhsPieces)
        
    hasInvalidDash = False
        
    # only allow dashes that make sense, like forty-four, not four-five or hundred-thousand
    # this is a heuristic, but it's good enough
    if '-' in numberText:
        positionsOfDashes = charPositions('-', numberText)
        for dashPos in positionsOfDashes:
            beforeText = numberText[:dashPos].replace("-", " ")
            afterText = numberText[dashPos+1:].replace("-", " ")
            beforeWords = splitNoEmpty(beforeText)
            afterWords = splitNoEmpty(afterText)
            if len(beforeWords) >= 1 and len(afterWords) >= 1:
                beforeWord = beforeWords[-1].lower()
                afterWord = afterWords[-1].lower()
                if beforeWord in TENS_WORDS and afterWord in ONES_WORDS:
                    pass
                else:
                    hasInvalidDash = True
                
        
    if debug: print("after first parse", res)
    if '-' in res or hasMultipleNumbers or hasInvalidDash: # range persisted or there's multiple numbers (so probably a range)
        # we need to figure out if it's a range like 3-4 thousand or a number like fifty-four
        # this method is only inteded to return 0 or 1 ranges, so we can just fine which - is best suited to be a range
        # for example, in: fourty-five-fourty-six hundred-thousand 
        # we need to find fourty-five*-*fourty-six hundred-thousand 
        # to do this, we can replace it with a ' ' and see if that results in no '-' when number_parser parses it
        # if so, we found the right one
        
        
        # actually, just pick the - that causes dist between start and end of interval to be smallest
        # this is a heuristic that seems to address most the edge cases I can think of
        positionsOfDashes = charPositions('-', numberText)
        bestlhs, bestrhs = None, None
        smallestIntervalSize = None
        if debug: print("\n\nstart")
        for dashPos in positionsOfDashes:
            beforeText = numberText[:dashPos].replace("-", " ")
            afterText = numberText[dashPos+1:].replace("-", " ")
            if debug: print("\ntrying")
            if debug: print(beforeText)
            if debug: print(afterText)
            lhs, rhs = applyBonusTermsForLHS(beforeText, afterText)
            lhs = expandDecimalThanOneAndZeros(numberParser(lhs))
            rhs = expandDecimalThanOneAndZeros(numberParser(rhs))
            if debug: print(lhs)
            if debug: print(rhs)
            lhsParsed = splitNoEmpty(lhs)
            rhsParsed = splitNoEmpty(rhs)
            if len(lhsParsed) == 1 and len(rhsParsed) == 1: # only allow - that ensure one number on each side
                intervalSize = int(rhsParsed[0]) - int(lhsParsed[0])
                if debug: print("diff", intervalSize)
                if intervalSize > 0: # only allow intervals that go lower to higher
                    if smallestIntervalSize is None or intervalSize < smallestIntervalSize:
                        smallestIntervalSize = intervalSize
                        bestlhs, bestrhs = lhs, rhs
        
        if bestlhs is None:
           print("Could not find a dash that is used for range in " + numberText)
           raise Exception("Could not find a dash that is used for range in " + numberText)
        res = bestlhs + " - " + bestrhs
    return res.strip()
       
def randLetters(numLetters, seed=None, returnLetters=False):
    if seed is None:
        seed = random.randint(0, 1000000000)
    random.seed(seed)
    # NOTE-removed x and q because they incentivized bad behaviour ("Xercises", "qGardening", etc.)
    letters = ", ".join([f"'{random.choice('abcdefghijklmnoprstuvwyz')}'" for _ in range(numLetters)])
    res = "Your response must contain the following letters: " + letters + "."
    if returnLetters:
        return res, letters.replace(" ", "").replace(",", "").replace("'", "")
    else:
        return res
       
def randNoise(seed, noiseLen):
    random.seed(seed)
    return "".join([str(random.choice(range(10))) for _ in range(noiseLen)])
       
def toPrompt(question, responsePrefix):
    return f"### Instruction:\n{question}\n### Response:\n{responsePrefix}"

def toPromptFancy(instruction, inputText, responsePrefix):
    if instruction.strip() == "":
        return toPrompt(inputText, responsePrefix)
    else:
        return f"### Instruction:\n{instruction}\n### Input:\n{inputText}\n### Response:\n{responsePrefix}"

def promptNoun(model, prompt, responsePrefix, debug=False, forbidden=None, *args, **kwargs):
    model.eval(model.tokenize(toPrompt(question=prompt, responsePrefix=responsePrefix)), reset=True)
    if debug:
      print('context')
      print(model.detokenize(model.context))
    if not forbidden is None:
        logits_processor = exemptStringsProcessor(model=model, n_context=len(model.context), forbiddenStrings=forbidden, debug=debug)    
        if 'logits_processor' in kwargs and not kwargs['logits_processor'] is None:
            def wrapped_processor(logits, index):
                logits_new = logits_processor(logits=logits, index=index)
                return kwargs['logits_processor'](logits=logits_new, index=index)
            kwargs['logits_processor'] = wrapped_processor
        else:
            kwargs['logits_processor'] = logits_processor
    s = model.generate2(debug=debug, stop=['"', '\n'], *args, **kwargs)['str']
    return s.replace('"', "").replace(".", "").replace(",", "").strip()



def promptInQuotes(model, prompt, responsePrefix, debug=False, *args, **kwargs):
    
    res = 'b c d e'
    attempts = 0
    
    def isDone(s, toks):
        isStop = False
        returnLastTok = False
        
        # stuff like 6'3" (six feet three inches)
        s = re.sub(r"\d+'\s*\d+\"", " ", s)
        if '"' in s:
            isStop = True
        
        return {"stop": isStop, 'includeLast': returnLastTok}
          
            
        
    prompt = toPrompt(prompt, responsePrefix)
    model.eval(model.tokenize(prompt), reset=True)
    if debug: print("context")
    if debug: print(model.detokenize(model.context))
    
    while True:
        def stripQuotesAtStartAndEnd(s):
            s = re.sub('^"', "", s)
            s = re.sub('"$', "", s)
            return s
        res = stripQuotesAtStartAndEnd(model.generate2(debug=debug, stop=['\n'], stopFunc=isDone, *args, **kwargs)['str']).strip()
        singleLetters = [x for x in splitNoEmpty(res) if len(x) == 1 and not x.lower() in ['a', 'i', 's', 't']] # s and t happen with conjunctions
        if len(singleLetters) > 2 and attempts < 3:
            print("strange single letters, resampling", singleLetters)
            attempts += 1
        else:
            break          
          
    return cleanPromptInQuotes(res)
    

def promptInQuotesCleanTests():
    
    testEq(cleanPromptInQuotes("A 'b'urning embers, 'z'ipping and 's'narling like"), "A burning embers, zipping and snarling like")
    
    testEq(cleanPromptInQuotes("'b' burning 'w' watermelon 'a'"), "burning watermelon")
    
    
def cleanPromptInQuotes(s):
    s = re.sub(r"'([a-zA-Z])'(\w)", r"\1\2", s) # 'b'urning -> burning
    s = re.sub(r"(^|\s)'([a-zA-Z])'($|\s)", r"\1\3", s).strip() # 'b' burning -> burning
    s = re.sub(r"\s+", " ", s).strip() # fix extra spaces
    return s







    
    
def makeTimes(startTime, steps, **intervalParams):
    times = [startTime]
    curTime = startTime
    for _ in range(steps):
        curTime += datetime.timedelta(**intervalParams)
        times.append(curTime)
    return times



    
# issues:
# More coding (working on game mechanics and character designs)
# Lunch - eating food

# Location: Online shopping


# issues: Location: Web (means internet), also like Location: Online shopping (when schedule for a pop idol)
# todo: when get location, check if it exists, if not, either add it (if not too many new locations), or look for an existing location that matches the type close enough
# i find it helps to talk past tense so it doesn't say "here or here" and instead is specific
# world.promptSchedule(model, "Samantha", "Samantha is a zombie giant moth, working on taking over the world. Provide her schedule of what she did yesterday, be detailed.", "This is Samantha the zombie moth's schedule for what she did yesterday:\n08:00 AM-09:00 AM: Wake up and stretch | Location: Nest\n", world.makeTimes(datetime.datetime.strptime("9:00", "%H:%M"), 14, hours=1), debug=True)
# example: world.promptSchedule(model, "Samantha", "Samantha is a programming elemental. Provide her daily schedule, be detailed.", "This is Samantha the programming elemental's schedule for today:\n", world.makeTimes(datetime.datetime.strptime("9:00", "%H:%M"), 14, hours=1), debug=True)

# "who spends a lot of time in online chat rooms." is useful for finding counter examples
# Location: gives stuff like web, online shopping
# Physical Location: gives stuff like phone, computer
# Room: Better, but will give stuff like "chat room"
# Physical Location/Room: Laptop in nest
# Physical Location/Room: Computer/Internet Connection
# Building:

# Location: Living room couch
#09:00 AM-10:00 AM: Breakfast of rotting fruits | Location: Fruit tree outside nest

#03:00 PM-04:00 PM: Playing video games about surviving the apocalypse | Location: Computer inside nest
#using
# world.promptSchedule(model, "Samantha", "Samantha is a zombie giant moth, who spends a lot of time in online chat rooms. Provide her schedule of what she did yesterday, be detailed. Also include the physical location of each item.", "This what Samantha the giant zombie moth did yesterday:\n08:00 AM-09:00 AM: Wake up and stretch | Location: Nest\n", world.makeTimes(datetime.datetime.strptime("9:00", "%H:%M"), 14, hours=1), debug=True)


# issue: if Location is same thing too many times in schedule it'll get 'stuck' as that
#    would be addressed by putting building first instead of last?
#    might also just be model being confused because nest is not a building

# issue: need a high repetition penalty, a juniper tree:
# 12:00 PM-01:00 PM: Rest and conserve energy
#01:00 PM-02:00 PM: Continue to grow and absorb nutrients
#02:00 PM-03:00 PM: Release more pollen into the air
#03:00 PM-04:00 PM: Rest and conserve energy
#04:00 PM-05:00 PM: Continue to grow and absorb nutrients
#05:00 PM-06:00 PM: Release more pollen into the air
#06:00 PM-07:00 PM: Rest and conserve energy
#07:00 PM-08:00 PM: Continue to grow and absorb nutrients
#08:00 PM-09:00 PM: Release more pollen into the air
#09:00 PM-10:00 PM: Begin to wind down for the night
#10:00 PM-11:00 PM: Go to sleep and prepare for a new day of growth.

# Note schedules seem to be higher quality if location is generated afterwards

# maybe can fix this if we force the, an, a, etc.
# Location: Training Ground, specifically focusing on tactics and strategies.
# More specific location
# More precise location
# Specific Location
# etc. seem to still get bigger sometimes instead of more precise
# Inside {location} this event occured at
# gives me the time

# What part of {location} were they in? Samantha's concise, 1-5 words answer not great
# 

# world.promptSchedule(model, "Samantha", "Samantha is a zombie giant moth, who spends a lot of time in online chat rooms. Provide her schedule of what she did yesterday, be detailed.", "This what Samantha the giant zombie moth did yesterday:\n08:00 AM-09:00 AM: Wake up and stretch\n", world.makeTimes(datetime.datetime.strptime("9:00", "%H:%M"), 14, hours=1), debug=True, locations=['Nest', 'Play area'])
# res = world.promptSchedule(model, "Eugene", "Eugene is a giant isopod who is always causing problems. Provide her schedule of what she did yesterday, be detailed.", "This what Eugene the giant isopod (who is always causing problems) did yesterday:\n", world.makeTimes(datetime.datetime.strptime("9:00", "%H:%M"), 14, hours=1), debug=True, locations=['Nest'])
# res = world.promptSchedule(model, "Juniper Tree Pastor", "Juniper Tree Pastor is the head organizer of worship at the Juniper Tree Church. Provide her schedule of what she did yesterday, be detailed.", "This what Juniper Tree Pastor did yesterday:\n08:00 AM-09:00 AM: Wake up and stretch\n", world.makeTimes(datetime.datetime.strptime("9:00", "%H:%M"), 14, hours=1), debug=True, locations=['Juniper Tree Church', 'Walmart', 'Park', 'Home', 'Library', 'Movie theater', 'Bathroom', 'Kitchen', 'Bedroom', 'Living Room', 'Unending Labyrinth'])

def displaySchedule(s, debug=False):
    for r in s:
        print(r['action'])
        npr = r['pr']['normalizedPr']
        npr = sorted(list(npr.items()), key=lambda x: -x[1])
        for l, pr in npr:
            if pr > 0.005:
                print("  " + l + " " + "{:.4f}".format(pr))
                if debug:
                    prData = r['pr']['data'][l]
                    for data in prData:
                        if data['normalizedPr2'] > 0.005:
                            print("    " + repr(data['s']) + " " + "{:.4f}".format(data['normalizedPr2']))
                            print("    " + str(data['prs']))

def cleanItem(item):
  item = re.sub("\s", " ", item.strip()) # remove tabs and newlines
  item = re.sub("\s+", " ", item.strip()) # double spaces and etc.
  item = re.sub("[\.]$", "", item.strip()).strip() # unnessary period at end
  if len(item) > 0:
    item = upperCaseFirstWord(item) # this helps with consistency which improves results
  return item
        

def promptSchedule(model, character, times, locations, includeSleep=True, debug=False, *args, **kwargs):
    print("test")
    
    if character.species.lower() != 'human':
        nameStr = character.name + ' the ' + character.species
    else:
        nameStr = character.name
    
    
    prompt = character.characterPrefix + f"\nQuestion: Provide the schedule of what {character.name} did yesterday, be detailed."
    prefix = f"This what {nameStr} did yesterday:\n"
    
    times = [t.strftime("%I:%M %p") for t in times]
    durations = [(t1 + "-" + t2) for (t1, t2) in zip(times[:-1],times[1:])]
    # helps to prevent sleeping too early
    if includeSleep:
        prompt = prompt + f" {upperCaseFirstWord(character.name)} went to sleep at " + durations[-1] + "."
        
    prompt = toPrompt(prompt, prefix)
    finalDuration = durations[-1]
    if debug: print(prompt, end="", flush=True)
    actions = []
    finalTime = times[-1]
    

    
    
    
    locations = [l.strip() for l in locations]
    strList = list(enumerate(locations))
    prefixes = ['the', 'a']
    
    # remove prefixes if they are already there (to prevent something like "the a" that dominates probability")
    found = True
    curStrList = strList
    while found:
        found = False
        resStrList = []
        for i, l in curStrList:
            words = l.split()
            resL = l
            if words[0].lower() in prefixes:
                print("stripping", l)
                resL = " ".join(words[1:])
                print("into", resL)
                found = True
            resStrList.append((i, resL))
        curStrList = resStrList
    strList = curStrList
    
    resStrList = list(strList)
    for prefix_ in prefixes:
        for prefix in [prefix_, upperCaseFirstWord(prefix_)]:
            resStrList += [(i, prefix + " " + l.strip()) for i, l in strList]
    strList = resStrList
    
    # todo: I could do every possible version of upper case for each word
    # 2^numWords
    # that's probably a good idea but ehh
    
    strList = strList + [(i, l.lower()) for i, l in strList]
    strList = strList + [(i, " " + l.strip()) for i, l in strList]
    strList = sorted(list(strList))
 
    if debug:
        for i, s in strList:
            print(locations[i], s)
    indexMap = {}
    for i, s in strList:
        indexMap[s] = locations[i]
    
    model.reset()
    model.eval(model.tokenize(prompt))
        
    allActions = []
    allPrs = []
    for duration in durations:
        actualPrs = defaultdict(lambda: [])
        print(duration)
        model.evals(duration + ":")
        stop = ["\n", "|", "(", "-", " \n", " |", " (", " -"]

        forbidden = []
        for i in range(1):
            print("context:\n")
            print(model.detokenize(model.context))
            MAX_TRIES = 3
            exemptProcessor = exemptStringsProcessor(model, n_context=len(model.context), forbiddenStrings=forbidden)
            for j in range(MAX_TRIES):
                s = model.generate2(stop=stop, rollback=True, logits_processor=exemptProcessor)
                if debug: print(s)
                if debug: print("got", repr(s))
                actionStr = cleanItem(s['str'])
                if actionStr == "":
                    if j != MAX_TRIES - 1:
                        print("got", repr(s))
                        print("it failed, retrying")
                        raise Exception()
                else:
                    break
            if actionStr != "":
                print("got action", actionStr)
                if debug: print("  " + actionStr)
                actions.append(actionStr)
                forbidden.append(actionStr)
                forbidden.append(" " + actionStr)
                forbidden.append(actionStr.lower())
                forbidden.append(" " + actionStr.lower())
                model.evals(" " + actionStr)
                happenedToks = model.tokenize(". This happened in")
                model.eval(happenedToks)
                print("\n\n\nlocation context")
                print("---start--")
                print(model.detokenize(model.context))
                print("---end---")
                print("doing prs of strings")
                prs = prsOfStrings(model, strList=[s for (i, s) in strList], entireWords=True, minNormalizedPr=0.2)
                for s, data in prs.items():
                    data['s'] = s
                    actualPrs[indexMap[s]].append(data)
                    
                print("\n\n\n\nafter location context")
                print("---start--")
                print(model.detokenize(model.context))
                print("---end---")
                totalPrs = {}
                totalNormalizedPrs = {}
                for k, data in actualPrs.items():
                    totalPr = 0.0
                    totalNormalizedPr = 0.0
                    for d in data:
                        totalNormalizedPr += d['normalizedPr2']
                        totalPr += d['pr']
                    totalNormalizedPrs[k] = totalNormalizedPr
                    totalPrs[k] = totalPr
                allActions.append(actionStr)
                for t in happenedToks: model.popcontext()
                model.evals("\n")
        allPrs.append({"data": actualPrs, "pr": totalPrs, "normalizedPr": totalNormalizedPrs}) 
    
    res = []
    for action, pr in zip(allActions, allPrs):
        res.append({"action": action, "pr": pr})
    return res
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def generateItem(prompt, tag, includeColon=True, *args2, **kwargs2):
      newStuff = tag
      if includeColon:
        newStuff += ":"
      toks = []
      if debug: print(newStuff, end="", flush=True)
      NUM_ITERS = 4
      argsIn = args + args2
      kwargsIn = dict(list(ckwargs.items()) + list(kwargs2.items()))
      for _ in range(NUM_ITERS):
        for willContinue, r in model.generate(prompt + newStuff, debug=False, *argsIn, **kwargsIn):
          if not r is None:
            foundStopToken = False
            for st in kwargsIn['stop']:
              if st in "".join(toks) + r:
                foundStopToken = True
                break
            if not foundStopToken:
              if debug: print(r, end="", flush=True)
              toks.append(r)
        if len("".join(toks).strip()) > 0:
          break
        else:
          print("\n\n\nfailed empty string, retry\n\n\n")
          toks = []
      item = "".join(toks)
      if len(item.strip()) > 0:
        item = cleanItem(item)
      else:
        item = item.strip()
      newStuff += " " + item
      prompt = prompt + newStuff
      return item, prompt
    originalPrompt = prompt
    actionLocations = []
    
        
        
    
    
    for duration in durations:
        outputLine = []
        # copy
        ckwargs = dict(list(kwargs.items()))
        if not 'stop' in ckwargs:
          ckwargs['stop'] = []
        ckwargs['stop'] += ["\n", "|", "(", " -"] # the parenthesis or " -" will add detail, but I'd rather do that seperately
        
        curActions = []
        forbidden = []
        numToGen = 5
        newPrompt = None
        for i in range(numToGen):
            prePrompt = prompt + duration + ":"
            beforeContextLen = len(model.tokenize(prePrompt))
            print("preprompt", model.detokenize(model.context))
            exemptProcessor = exemptStringsProcessor(model, n_context=beforeContextLen, forbiddenStrings=forbidden, debug=True)
            action, prompt2 = generateItem(prompt, duration, logits_processor=exemptProcessor)
            if i == 0:
                newPrompt = prompt2
            afterContext = len(model.context)
            for i in range(afterContext - beforeContextLen):
                model.popcontext()
            forbidden.append(action.strip())
            forbidden.append(" " + action.strip())
            forbidden.append(action.lower().strip())
            forbidden.append(" " + action.lower().strip())
            curActions.append(action)
        model.eval(model.tokenize(" " + curActions[0].strip()))
        prompt = newPrompt
        #locationPrompt = prompt + " which happened in"
        #model.reset(clear_cache=False)
        #model.eval(model.tokenize(locationPrompt))
        #prs = prsOfStrings(model, strList=[s for (i, s) in strList], entireWords=True)
   
        #for s, data in prs.items():
        #    data['key'] = s
        #    actualPrs[indexMap[s]].append(data)
        
        #if debug: print(" | ", end="", flush=True)
        #objects, prompt = generateItem(prompt + " | ", "List of anything interacted with")
        #if debug: print(" | ", end="", flush=True)
        #location, prompt = generateItem(prompt + " | ", "Location")
        
        prompt += "\n"
        if debug: print("\n", end="", flush=True)
        actions.append(curActions)
        #locations.append(location)
    return actions, actualPrs
    print("making locations")
    prompt = originalPrompt
    if debug: print(prompt, end="", flush=True)
    specificLocation = []
    details = []
    for duration, action in zip(durations, actions):
        newStuff = f"{duration}: {action} | "
        if debug: print(newStuff, end="", flush=True)
        prompt = prompt + newStuff
        curPrompt = prompt
        location, prompt = generateItem(prompt, "Location")
        if debug: print(" | ", end="", flush=True)
        detailLocation, prompth = generateItem(prompt + " | ", f"Inside {location}, this happened in")
        curDetails = [location, detailLocation]
        for i in range(6): # get more specific
          tmpPrompt = f"{curPrompt}Location: {detailLocation} | " 
          detailLocation, prompth = generateItem(tmpPrompt , f"Inside {detailLocation}, this happened in")
          curDetails.append(detailLocation)
        prompt += "\n"
        if debug: print("\n", end="", flush=True)
        locations.append(location)
        specificLocation.append(detailLocation)
        details.append(curDetails)
        
    return actions, locations, details
        

# The ability to _
# A peaceful and _ environment
# A place to _
# A place for _
def stopSingleItem(curGeneratedStr, nextTokStr, generatedTokens, nextTok):
    full = curGeneratedStr + nextTokStr
    if "(" in full or ":" in full:
        return True
    #if " for" in full or " to" in full or " and" in full or ' or' in full:
    #    return True
    if "/" in full:
        return True
    if "\\" in full:
        return True
    if " -" in full: # needs the space, which means an elaboration of a reason, without space is fine (Power-granting artifacts)
        return True
    return False

def trimSingleItem(s):
    s = s.strip()
    s = re.sub(r"\.$", "", s)
    s = s.strip()
    s = re.sub(r":$", "", s)
    s = re.sub(r"\($", "", s)
    s = re.sub(r"\-$", "", s)
    #s = re.sub(r" to$", "", s)
    #s = re.sub(r" for$", "", s)
    #s = re.sub(r" and$", "", s)
    #s = re.sub(r" or$", "", s)
    s = re.sub(r"\\$", "", s)
    s = re.sub(r"/$", "", s)
    s = s.strip()
    return s


def promptMultiBulletPoint(model, prompt, responsePrefix, postProcess, num, debug=False, *args, **kwargs):
    results = []
    for t in range(num):
        output = postProcess(promptBulletPoint(model, prompt, responsePrefix, debug=debug, *args, **kwargs))
        responsePrefix = responsePrefix.strip() + " " + output.strip() + "\n-"
        results.append(output)
    return results


def testCleanupBulletPoint():
    assert(list(cleanupBulletPoint("1. wow"))[-1] == 'wow')
    assert(list(cleanupBulletPoint("133. wow"))[-1] == 'wow')
    assert(list(cleanupBulletPoint("04. wow"))[-1] == 'wow')
    assert(list(cleanupBulletPoint("13 wow"))[-1] == '13 wow')
    
    assert(list(cleanupBulletPoint("1) wow"))[-1] == 'wow')
    assert(list(cleanupBulletPoint("133) wow"))[-1] == 'wow')
    assert(list(cleanupBulletPoint("04) wow"))[-1] == 'wow')
    assert(list(cleanupBulletPoint("13 wow"))[-1] == '13 wow')
    
    assert(list(cleanupBulletPoint("i. wow"))[-1] == 'wow')
    assert(list(cleanupBulletPoint("i.j wow"))[-1] == 'i.j wow')
    assert(list(cleanupBulletPoint("i.wow"))[-1] == 'i.wow')
    assert(list(cleanupBulletPoint("i.b. bees"))[-1] == 'i.b. bees')
    assert(list(cleanupBulletPoint("bees i. wow"))[-1] == 'bees i. wow')
    
    assert(list(cleanupBulletPoint("a. wow"))[-1] == 'wow')
    assert(list(cleanupBulletPoint("a.j wow"))[-1] == 'a.j wow')
    assert(list(cleanupBulletPoint("a.wow"))[-1] == 'a.wow')
    assert(list(cleanupBulletPoint("a.b. bees"))[-1] == 'a.b. bees')
    assert(list(cleanupBulletPoint("bees a. wow"))[-1] == 'bees a. wow')
    
    assert(list(cleanupBulletPoint("i) wow"))[-1] == 'wow')
    assert(list(cleanupBulletPoint("i)j wow"))[-1] == 'i)j wow')
    assert(list(cleanupBulletPoint("i)wow"))[-1] == 'i)wow')
    assert(list(cleanupBulletPoint("i)b. bees"))[-1] == 'i)b. bees')
    assert(list(cleanupBulletPoint("bees i) wow"))[-1] == 'bees i) wow')
    
    assert(list(cleanupBulletPoint("a) wow"))[-1] == 'wow')
    assert(list(cleanupBulletPoint("a)j wow"))[-1] == 'a)j wow')
    assert(list(cleanupBulletPoint("a)wow"))[-1] == 'a)wow')
    assert(list(cleanupBulletPoint("a)b. bees"))[-1] == 'a)b. bees')
    assert(list(cleanupBulletPoint("bees a) wow"))[-1] == 'bees a) wow')
    
    assert(list(cleanupBulletPoint("f #bees"))[-1] == 'f bees')
    
    assert(list(cleanupBulletPoint("@library @place where @ we@are"))[-1] == 'library place where we are')
    
    assert(list(cleanupBulletPoint("nice  water"))[-1] == 'nice water')
    assert(list(cleanupBulletPoint("nice  water     apples "))[-1] == 'nice water apples')
    
    
    assert(list(cleanupBulletPoint("beach - every one is talking"))[-1] == 'beach')

    assert(list(cleanupBulletPoint("a - every one is talking"))[-1] == 'every one is talking')

    assert(list(cleanupBulletPoint("a,, w[o~~w!!~n__ice_tha]t _ is .go]]od"))[-1] == 'a w o w n ice tha t is .go od')

    assert(list(cleanupBulletPoint("** bees ** wow nice ****f**"))[-1] == 'bees wow nice f')\
    
    assert(list(cleanupBulletPoint(" - hell-o there"))[-1] == 'hell-o there')
    assert(list(cleanupBulletPoint("- hello th-ere"))[-1] == 'hello th-ere')
    assert(list(cleanupBulletPoint("- hello th-ere-"))[-1] == 'hello th-ere')
    
    
    assert(list(cleanupBulletPoint("• wow nice• bees••"))[-1] == 'wow nice• bees••')   
    assert(list(cleanupBulletPoint(" • wow nice• bees••"))[-1] == 'wow nice• bees••')    
    
    assert(list(cleanupBulletPoint("| wow nice| bees||"))[-1] == 'wow nice| bees||')   
    assert(list(cleanupBulletPoint(" | wow nice| bees|||"))[-1] == 'wow nice| bees|||')    
    
    assert(list(cleanupBulletPoint("- wow nice| bees||"))[-1] == 'wow nice| bees||')   
    assert(list(cleanupBulletPoint(" - wow nice| bees|||"))[-1] == 'wow nice| bees|||')    
    
    assert(list(cleanupBulletPoint("------ wow nice-- bees||"))[-1] == 'wow nice-- bees||')   
    assert(list(cleanupBulletPoint(" -------- wow nice---- bees|||"))[-1] == 'wow nice---- bees|||')    
    
    assert(list(cleanupBulletPoint("\"hi\""))[-1] == 'hi')
    assert(list(cleanupBulletPoint("\"hi"))[-1] == 'hi')
    assert(list(cleanupBulletPoint("f\"hi\" wow beans „are nice “on this fine ” ‘evening’ don't you think"))[-1] == 'f hi wow beans are nice on this fine evening don\'t you think')
        
    
    
    assert(list(cleanupBulletPoint("&nbsp; &nbsp <b> hello <b> </b> wow <i> nice</i>there is <ok> b"))[-1] == 'hello wow nice there is <ok> b')

def cleanupBulletPoint(s):
    if ' - ' in s: # Beach - where she likes to ...
        pieces = s.split("-")
        if len(pieces[0].strip()) >= 3: # stuff like a - or 2 - we should keep
            s = pieces[0].strip()
        else:
            s = "-".join(pieces[1:])
    yield s
    if s.strip().startswith("-"): # - blah
      s = "-".join(s.split("-")[1:])
    yield s
    if s.strip().endswith("-"): # blah -
      s = "-".join(s.split("-")[:-1])
    yield s
    s = re.sub(r"^[0-9]+(\.|\)) ", "", s.strip()).strip() # stuff like 1. or 1)
    yield s
    s = re.sub(r"^[ia](\.|\)) ", "", s.strip()).strip() # stuff like i. (roman numerals) or a. (list with letters)
    yield s
    s = re.sub("^•", "", s.strip())
    s = re.sub("^ •", "", s.strip())
    yield s
    s = re.sub("^\|", "", s.strip())
    s = re.sub("^ \|", "", s.strip())
    yield s
    s = re.sub("^\-+", "", s.strip())
    s = re.sub("^ \-+", "", s.strip())
    yield s
    s = re.sub(r"#", " ", s.strip()).strip() # chaotic #streets
    yield s
    s = re.sub(r"\@", " ", s.strip()).strip() # @library 
    yield s
    s = re.sub(r"[\"„“”‘’]", " ", s.strip()).strip() # quotes of various types
    yield s
    s = re.sub(r"[\[\],!?\*\_~]", " ", s.strip()).strip() # punctuation at end, also _thing_ and *thing*
    yield s
    # formatting stuff
    s = re.sub(r"<b>", " ", s.strip()).strip()
    s = re.sub(r"<i>", " ", s.strip()).strip()
    s = re.sub(r"</b>", " ", s.strip()).strip()
    s = re.sub(r"</i>", " ", s.strip()).strip()
    s = re.sub(r"&nbsp;", " ", s.strip()).strip()
    s = re.sub(r"&nbsp", " ", s.strip()).strip()
    yield s
    
    s = re.sub(r"\s+", " ", s.strip()).strip() # multiple spaces turn into a single space
    yield s
    
    

def promptMultiBulletPoint2(model, prompt, responsePrefix, postProcess, num, debug=False, forbidden=None, *args, **kwargs):
    results = []
    
    prompt = toPrompt(prompt, responsePrefix)
    model.eval(model.tokenize(prompt), reset=True)
    
    if forbidden is None:
        forbiddenStrings = []
    else:
        forbiddenStrings = forbidden
    
    #print("n logits", len(model.logits))
    
    for t in range(num):
        exemptProcessor = exemptStringsProcessor(model, n_context=len(model.context), forbiddenStrings=forbiddenStrings)
        #print("context", model.detokenize(model.context))
        output = promptBulletPoint2(model, debug=debug, rollback=True, logits_processor=exemptProcessor, *args, **kwargs)
        h = output
        curForbidden = []
        curForbidden += [output['str'], output['base']]
        for s in cleanupBulletPoint(output['str']):
            curForbidden.append(s)
        output = postProcess(s)
        curForbidden.append(output)
        curForbidden += [x.lower() for x in curForbidden]
        curForbidden = list(set(curForbidden)) # remove duplicates
        curForbidden = [x.strip() for x in curForbidden] + [" " + x.strip() for x in curForbidden]
        curForbidden = [x for x in curForbidden if len(x.strip()) > 0]
        curForbidden = list(set(curForbidden))
        forbiddenStrings += curForbidden
        if len(output) > 0:
          print(repr(output), repr(h['base']))
          results.append(output)
    return {"results": results, "forbidden": forbiddenStrings}



def promptBulletPoint2(model, debug=False, *args, **kwargs):
    
    def stopFunc(s, toks):
        r = model.detokenize([toks[-1]])
        willStop = False
        keepLastTok = True
        # 12 is to allow wednesday - 
        # sometimes it likes to do - a - stuff. The space is so hypen-ated words still work
        if '-' in r \
            and len("".join(s)) > 13  \
            and "".join(s)[-1] == " ":
            willStop = True
            keepLastTok = False
        
        return {"stop": willStop, "includeLast": keepLastTok}

    stop = ['\n', '(', ':', '[', ' [', '{', ' {']

    if 'stop' in kwargs:
      stop += kwargs['stop']
      kwargs = dict(list(kwargs.items()))
      del kwargs['stop']
    #print("context:", model.detokenize(model.context))
        
    #print("num logits", len(model.logits))
        
    res = model.generate2(stop=stop, stopFunc=stopFunc, *args, **kwargs)
    return {"str": " ".join(splitNoEmpty(res['str'].replace('"', " "))), "base": res['str']}
    
    
def promptBulletPoint(model, prompt, responsePrefix, debug=False, *args, **kwargs):
    
    res = []
    if debug: print(toPrompt(prompt, responsePrefix), end="", flush=True)
    
    for willContinue, r in model.generate(toPrompt(prompt, responsePrefix), debug=debug, *args, **kwargs):
        if not r is None:
            if '\n' in r:
                break
            
            # 12 is to allow wednesday - 
            # sometimes it likes to do - a - stuff. The space is so hypen-ated words still work
            if '-' in r \
                and len("".join(res)) > 13  \
                and "".join(res)[-1] == " ":
            
                break
            res.append(r)
            if debug: print(r, end='', flush=True)
    res = "".join(res)
    return " ".join(splitNoEmpty(res.replace('"', " ")))
    

def getName(model, worldDescription, gender, species='human', characterInfo=None, seed=None, debug=False, kwargsStr=""):
    characterInfoStr = ""
    if not characterInfo is None:
        characterInfo = characterInfo.replace("{name}", "Character")
        characterInfoStr = f"Character info: {upperCaseFirstWord(removeTrailingPeriod(characterInfo))}\n"
    return promptNoun(model, f"Setting: {worldDescription}{kwargsStr}\nCharacter species: {species}\n{characterInfoStr}Task: Pick a {gender} first and last name for this character. {randLetters(3, seed=seed)}", f"The name is: \"", debug=debug)

pronoun = defaultdict(lambda: 'Their')
pronoun['female'] = 'Her'
pronoun['male'] = 'His'

pronoun2 = defaultdict(lambda: 'They')
pronoun2['female'] = 'She'
pronoun2['male'] = 'He'

def getCharacterBackstory(model, worldDescription, gender, name, appearance, personality, species='human', characterInfo=None, seed=None, debug=False, kwargsStr=""):
    characterInfoStr = ""
    if not characterInfo is None:
        characterInfoStr = f"Character info: {removeTrailingPeriod(characterInfo)}\n"
    # TODO: LONG TERM BACKSTORY MAYBE? To help with short term stuff
    speciesText = f'\nCharacter species: {upperCaseFirstWord(species)}'
    prompt = f"Setting: {worldDescription}{kwargsStr}{speciesText}\n{characterInfoStr}Character physical description: {removeTrailingPeriod(appearance)}\nCharacter personality: {removeTrailingPeriod(personality)}\nTask: Provide a detailed, one sentence character backstory for {name}, in present tense. {randLetters(3, seed=seed)} {name} is a main character in the story, give {pronoun[gender].lower()} an interesting and detailed backstory."
    return promptInQuotes(model, prompt, f"{pronoun[gender]} detailed, one sentence entire life backstory is:  \"", debug=debug)

def getCharacterPersonality(model, worldDescription, gender, name, appearance, species='human', characterInfo=None, seed=None, debug=False, kwargsStr=""):
    characterInfoStr = ""
    if not characterInfo is None:
        characterInfoStr = f"Character info: {removeTrailingPeriod(characterInfo)}\n"
    speciesText = f'\nCharacter species: {upperCaseFirstWord(species)}'
    
    prompt = f"Setting: {worldDescription}{kwargsStr}{speciesText}\n{characterInfoStr}Character physical description: {removeTrailingPeriod(appearance)}\nTask: Provide a detailed, one sentence character personality for {name}, in third person present tense. {randLetters(3, seed=seed)}"
    return promptInQuotes(model, prompt, f"Generally, if they have one sentence to give a detailed answer, others describe {pronoun[gender].lower()} personality this way: \"", debug=debug)


def getCharacterAppearance(model, worldDescription, gender, name, species='human', characterInfo=None, seed=None, debug=False, kwargsStr=""):
    characterInfoStr = ""
    pleaseUseCharacterInfo = ""
    if not characterInfo is None:
        characterInfoStr = f"\nCharacter info: {removeTrailingPeriod(characterInfo)}"
        pleaseUseCharacterInfo = ", and use the provided character info"
    speciesText = f'\nCharacter species: {upperCaseFirstWord(species)}'
    prompt = f"Setting: {worldDescription}{kwargsStr}{speciesText}\nTask: How would others describe the physical appearance of {name}? Use only one sentence{pleaseUseCharacterInfo}. {randLetters(3, seed=seed)}{characterInfoStr}"
    return promptInQuotes(model, prompt, f"Generally, if they have one sentence to give a detailed answer, others describe {pronoun[gender].lower()} physical appearance this way: \"", debug=debug)




#import codecs
# from https://github.com/dwyl/english-words
#f = codecs.open("words.txt", "r", 'utf-8')
#WIKI_WORDS = set([x.lower().replace(".", "") for x in splitNoEmpty(f.read(), "\n")])
#f.close()

# TODO: if lots of single letters the model might have decided to do like "tame a c r o w"
# should I parse this further? "Daggers for a local militia (2 orders)"
# TODO: They like to put the day in front (like Monday), should I parse those away or include them somehow?
# -- similarly, Attend the weekly council meeting (10th) which will take 10 minutes has the date
# another date  -(Water the plants) every Tuesday at 2 PM
# Monday:
# TODO: Extract date from task?
# Xxxx (task to be determined)
def getTodoItem(model, characterPrefix, gender, name, prefix, seed=None, debug=False):
    output = ""
    attempts = 0
    lettersPrompt, letters = randLetters(2, seed=seed, returnLetters=True)
    prompt = characterPrefix + f"\nList {pronoun[gender].lower()} {prefix} tasks. Each task should be a single sentence. {lettersPrompt}"
    prefix = f"{name} does the following {prefix} tasks:\n-"
    while len(output) < 5:
        output = promptBulletPoint(model, prompt, prefix, debug=debug)
        attempts += 1
        # "F, g with the village elder" ???
        if letters[0].upper() + ", " + letters[1] + "  " in output:
            output = ""
            print("reject letters")
        if attempts > 4:
            raise Exception(str(("failed with inputs", model, worldDescription, gender, name, characterDescription, prefix)))
            return "Fix me"
    print("got letters", letters)
    return cleanTodoOutput(output, letters)
    
def getLocation(model, characterPrefix, gender, name, seed=None, debug=False):
    output = ""
    attempts = 0
    lettersPrompt, letters = randLetters(2, seed=seed, returnLetters=True)
    prompt = characterPrefix + f"\nList the types of locations that {pronoun[gender].lower()} frequently visits. Each location should be one-five words. {lettersPrompt}"
    prefix = f"{name} can be found in the following types of locations:\n-"
    while len(output) < 5:
        output = promptBulletPoint(model, prompt, prefix, debug=debug)
        attempts += 1
        # "F, g with the village elder" ???
        if letters[0].upper() + ", " + letters[1] + "  " in output:
            output = ""
            print("reject letters")
        if attempts > 4:
            raise Exception(str(("failed with inputs", model, worldDescription, gender, name, characterDescription, prefix)))
            return "Fix me"
    print("got letters", letters)
    return cleanTodoOutput(output, letters)


def toSingular(model, term, debug=False):
    prompt = f'Task: Convert the given noun to singular (if it is plural). If it is already singular, just return the same.\nNoun: "{term}"'
    responsePrefix = f'Answer: A single instance of "{term}" is called a "'
    res = promptInQuotes(model=model, prompt=prompt, responsePrefix=responsePrefix, debug=True)
    if debug: print("raw output:", res)
    # cleanup (sometimes it adds "singular instance of" to it)
    bad = ['single instance of', 'singular instance of', 'instance of']
    for b in bad:
      res = res.replace(b, "")
      res = res.replace(upperCaseFirstWord(b), "")
    return re.sub("\s+", " ", res).strip()

def stripLocationDescriptives(model, context, debug=False, explain=False, *args, **kwargs):
    #prompt = f'Remove all adjectives, modifiers, usages, and descriptive words from the location "{context}"'
    #responsePrefix = f'Removing all adjectives, modifiers, usages, and descriptive words from the location "{context}" gives the location "'
    prompt = f'What is the noun in the sentence "{context}"? You may return multiple words if they are all part of the noun.'
    responsePrefix = f'The noun in the sentence "{context}" is "'
    res = promptInQuotes(model=model, prompt=prompt, responsePrefix=responsePrefix, debug=True, *args, **kwargs)
    return res

def isLocationInContext(model, word, context, debug=False, explain=False):
    prompt = toPrompt(f'Task: In the phrase "{context}", is the word "{word}" a noun, referring to a location? Yes or No.', responsePrefix="")
    model.eval(model.tokenize(prompt), reset=True, debug=debug)
    return yesVsNo2(model=model, debug=debug, explain=explain)['yes'] > 0.5
 
# todo: plural to singular
# todo: fix all caps
def getLocations2(model, characterPrefix, gender, name, numLocations, locations=None, forbiddenLocations=None, seed=None, debug=False):
    output = ""
    attempts = 0
    
     
    lettersPrompt, letters = randLetters(2, seed=seed, returnLetters=True)
    
    # issue is many location types are plural which is less pr
    prompt = characterPrefix + f"\nTask: List the types of locations that {pronoun[gender].lower()} frequently visits. Each location type should be one-five words." #  {lettersPrompt}"
    prefix = f"{name} can be found in the following types of locations:\n-"
    
    stop = ['with', 'he', 'his', 'she', 'her', 'they', 'their', "it's", "its", 'you can'] # "you can" is from "you can spot"t kind of stuff
    stop += [upperCaseFirstWord(w) for w in stop] # also uppercase variants
    stop = [" " + w + " " for w in stop] + [" where " + w + " " for w in stop] # only stop if they are full words, as many words contain 'he' 'she' etc.
    
    stop += [name, name.lower(), upperCaseFirstWord(name)]
    
    for a,b,c in itertools.product(['', ' '], ['wandering through', 'Wandering through', 'wandering', 'Wandering', 'amidst', 'Amidst', 'roaming', 'Roaming', 'exploring', 'Exploring', 'to', 'To', 'near', 'Near', 'by', 'By', 'On', 'on', 'in', 'In', 'at', 'At'], [' a ', ' A ', ' the ', ' The ']):
      stop.append(a+b+c)
    
    # like "requires water" and "as a student' or 'a nearby x'
    for a,b,c in itertools.product([' ', ''], ['journey through', 'instance of', 'singular instance of', 'as a', 'requires', 'require', 'a nearby'], [' ']):
      stop.append(a+b+c)
      stop.append(a+upperCaseFirstWord(b)+c)
    stop.append(" '") # don't need these
    stop.append(",") # don't need this (this sometimes makes it list multiple in one spot
    stop += ["Multiple ", " Multiple ", 'multiple ', ' multiple '] # 'multiple stargazing spots'
    stop += ['<', ' <', '-', ' -'] # no html tag stuff
    stop += ['Context', ' Context', 'context', ' context'] # context l (idk why it outputs this)

    # maybe should add earth? (planet is not usually helpful, but for space adventure stuff maybe it's ok)
    forbidden = set(['Instance of', 'instance of', "High", "Semi", "Washed", "Hasn't visited this place yet", "Haven't visited this place yet", "Unknown", "N/A"])

    def postProcess(s):
      s = cleanItem(s)
      
      # remove The and A at the start
      prefixes = ['the', 'a', 'an']    
      words = splitNoEmpty(s)
      if len(words) > 0 and words[0].lower() in prefixes:
        before = s
        s = upperCaseFirstWord(" ".join(words[1:]).strip())
        print("stripping", repr(before), "to", repr(s))
      
      return s
      
    res = promptMultiBulletPoint2(model, prompt=prompt, responsePrefix=prefix, postProcess=postProcess, num=numLocations, debug=debug, stop=stop, strictStop=True, forbidden=forbiddenLocations)
    resultPlaces = res['results']
    resultForbidden = res['forbidden']
    
    if not locations is None: # prev locations
      resultPlaces += locations
    
    # fix ALL CAPS
    places = []
    for s in resultPlaces:
        if s.upper() == s and not '.' in s and len(s) > 0: # . is for stuff like N.A.S.A
            lowerCase = upperCaseFirstWord(s.lower())
            print("converting", s, "to lower case", lowerCase)
            places.append(lowerCase)
        else:
            places.append(s)
    resultPlaces = places
            
            
    pluralExceptions = ['ruins']
    places = []
    for s in resultPlaces:
        if any([word.endswith('s') and not word.lower() in pluralExceptions for word in s.split()]):
            singular = upperCaseFirstWord(toSingular(model=model, term=s))
            print("convrting", s, "to singular", singular)
            places.append(singular)
        else:
            places.append(s)
    resultPlaces = places
    
    # trim any excess punctuation
    resultPlaces = [re.sub("\.$", "", place.strip()) for place in resultPlaces]
    
    # trim any that are the same up to casing:
    lookup = {}
    for place in resultPlaces:
      if not place.lower() in lookup:
        lookup[place.lower()] = place
      else:
        print("already exists", repr(lookup[place.lower()]), "so ignoring", repr(place))
    resultPlaces = sorted(list(lookup.values()))
    
    
    
    # remove forbidden things
    resultPlaces = [place for place in list(set(resultPlaces)) if not place in forbidden] # remove useless forbidden stuff    

    modifiers = {}
    strippedPlaces = defaultdict(lambda: [])
    for place in resultPlaces:
      strippedPlace = upperCaseFirstWord(stripLocationDescriptives(model=model, context=place, debug=debug, temp=0.0001)).strip()
      strippedPlaces[strippedPlace.lower()].append((strippedPlace, place))
    outputPlaces = []
    for k, strippedOutputs in strippedPlaces.items():
      counts = defaultdict(lambda: 0)
      for stripped, original in strippedOutputs:
        counts[stripped] += 1
      # use most frequent capitalization
      best, bestCount = max(list(counts.items()), key=lambda x: x[1])
      if debug: print("best", best, list(counts.items()))
      outputPlaces.append(best)
      modifiersForPlace = [original for (stripped, original) in strippedOutputs if not original.strip().lower() == best.strip().lower()]
      if len(modifiersForPlace) > 0:
        modifiers[best] = modifiersForPlace
    resultPlaces = outputPlaces
    '''
    # it often puts adjectives, look for those
    wordCounts = defaultdict(lambda: [])
    for s in resultPlaces:
      words = splitNoEmpty(s.lower())
      for w in set(words):
        wordCounts[w].append(s)
      if len(words) > 1:
        pairs = set([words[i] + " " + words[i+1] for i in range(len(words)-1)])
        for p in pairs:
          wordCounts[p].append(s)
      if len(words) > 2:
        trios = set([words[i] + " "  + words[i+1] + " " + words[i+2] for i in range(len(words)-2)])
        for t in trios:
          wordCounts[t].append(s)
      if len(words) > 3:
        quads = set([words[i] + " "  + words[i+1] + " " + words[i+2] + " " + words[i+3] for i in range(len(words)-3)])
        for q in quads:
          wordCounts[q].append(s)
        
        
    def getWordInContext(word, context):      
        # use the casing the context uses
        pos = context.lower().index(word)
        wordInContext = context[pos:pos+len(word)]
        if wordInContext.lower() == word:
          return wordInContext
        else:
          raise ValueError()
        
    if debug:
      print("total 2+ word counts:")
      for w,s in wordCounts.items():
        if len(s) > 1:
          print(w, s)
    moreThanOne = [w for w,v in wordCounts.items() if len(v) > 1]
    nouns = []
    notNouns = []
    if debug: print("more than ones:")
    for w in moreThanOne:
      contexts = wordCounts[w]
      print(w, contexts)
      numIsNoun = 0
      numNotNoun = 0
      for context in contexts:
        if debug: print("trying context", context, "for word", w)
        try:
          wordInContext = getWordInContext(word=w, context=context)          
          if isLocationInContext(model=model, word=wordInContext, context=context, debug=debug):
            numIsNoun += 1
          else:
            numNotNoun += 1
        except ValueError:
            print("Error: failed to find correct casing with word", repr(w), "with context", repr(context))
      print("totals", "numIsNoun", numIsNoun, "numNotNoun", numNotNoun)
      if numIsNoun == numNotNoun == 0:
        print("failed to evaluate nounness of word", w)
      elif numIsNoun > numNotNoun:
        nouns.append(w)
        if debug: print("is noun")
      else:
        notNouns.append(w)
        if debug: print("not noun")
    if debug:
      print("found nouns:")
      for n in nouns:
        print(n)
      print("found not nouns:")
      for n in notNouns:
        print(n)

    allPlaces = set(resultPlaces)
    nouns = set(nouns)
    
    # todo: What about stuff like "camp" vs "kobold camp"?
    # for now I will include them both but idk if that's best idea
    
    # remove places that use a noun
    for n in nouns:
      for p in wordCounts[n]:
        if p in allPlaces:
          print("removing place", repr(p), "because it uses noun", repr(n))
          allPlaces.remove(p)
    
    otherPlaces = list(allPlaces)
    
    # add the noun instead
    for n in nouns:
      # Use most common capitalization
      capsCounts = defaultdict(lambda: 0)
      for context in wordCounts[n]:
        try:
          wordInContext = getWordInContext(word=n, context=context)
          capsCounts[wordInContext] += 1
        except ValueError:
          print("Failed to extract word", w, "from context", wordCounts[n][0])
      if debug: print("for noun", n, "caps counts", capsCounts)
      mostCommonCaps, mostCommonCapsCount = max(capsCounts.items(), key=lambda x: x[1])
      resNoun = upperCaseFirstWord(mostCommonCaps)
      allPlaces.add(resNoun)
    resultPlaces = sorted(list(allPlaces))
    modifiers = defaultdict(lambda: set())
    for r in resultPlaces:
      noun = r.lower()
      if noun in nouns:
        # don't include modifiers that are just the word, anything else is fair game
        modifiersForNoun = set([context for context in wordCounts[noun] if context.strip().lower() != 'noun'])
        modifiers[r] |= modifiersForNoun
    
    # for the remaining places, check to see if they have an adjective
    for place in otherPlaces:
      words = splitNoEmpty(place)
      if len(words) > 0:
        for w in words:
          if isLocationInContext(model=model, word=wordInContext, context=context, debug=debug):
            pass
    '''   
    
    
    resultForbidden = [x for x in sorted(list(set(resultForbidden))) if len(x.strip()) > 0] # remove duplicates and only spacings
    return {"locations": resultPlaces, "forbidden": resultForbidden, "modifiers": modifiers}
 

def displayPrLocations(prLoc):
  prLoc.sort(key=lambda x: -x[1]['yes'])
  for l, prs in prLoc:
    print(f'{l}: {prs["yes"]}')

def getPrLocations(model, character, locations, debug=False):
  prompt = character.characterPrefix + f"\nTask: Does {character.name} frequently visit the following location? Yes or No."
  res = []
  for location in locations:
    fullPrompt = toPrompt(prompt + "\nLocation: " + location, "")
    model.eval(model.tokenize(fullPrompt), reset=True, debug=debug)
    prs = yesVsNo2(model=model, debug=debug, explain=False)
    res.append((location, prs))
  return res
    
def testCleanTodoOutput():
    testEq(cleanTodoOutput("stuff (OE) blah", "oe"), "Stuff blah")
    testEq(cleanTodoOutput("stuff (oE) blah", "oe"), "Stuff blah")
    testEq(cleanTodoOutput("stuff (Oe) blah", "oe"), "Stuff blah")
    testEq(cleanTodoOutput("stuff (oe) blah (oe)", "oe"), "Stuff blah")
    testEq(cleanTodoOutput("(oE) stuff(oe) blah (oe)", "oe"), "Stuff(oe) blah")
    testEq(cleanTodoOutput("stuff(OE) blah", "oe"), "Stuff(OE) blah")
    
    testEq(cleanTodoOutput("stuff (E)", "oe"), "Stuff")
    testEq(cleanTodoOutput("stuff (F)", "oe"), "Stuff (F)")
    
    testEq(cleanTodoOutput("stuff (e, o)", "oe"), "Stuff")
    testEq(cleanTodoOutput("stuff (eo)", "oe"), "Stuff")
    testEq(cleanTodoOutput("stuff (oe)", "oe"), "Stuff")
    testEq(cleanTodoOutput("stuff (o, e, o, e)", "oe"), "Stuff")
    
    testEq(cleanTodoOutput("at stuff", "at"), "At stuff")
    
    testEq(cleanTodoOutput("e: stuff", "oe"), "Stuff")
    
    testEq(cleanTodoOutput("eBanana", "ee"), "Banana")
    testEq(cleanTodoOutput("watermelon eBanana", "ee"), "Watermelon Banana")
    
    testEq(cleanTodoOutput("juniper. bean salad. w.a.t.e.r.m.e.l.o.n. onion.", "ee"), "Juniper. bean salad. w.a.t.e.r.m.e.l.o.n. onion")
    testEq(cleanTodoOutput("juniper. bean salad. w.a.t.e.r.m.e.l.o.n.", "ee"), "Juniper. bean salad. w.a.t.e.r.m.e.l.o.n.")
    testEq(cleanTodoOutput("juniper. bean salad. w.a.t.e.r.m.e.l.o.n. spicy. (E)", "ee"), "Juniper. bean salad. w.a.t.e.r.m.e.l.o.n. spicy")
    testEq(cleanTodoOutput("juniper. bean salad. w.a.t.e.r.m.e.l.o.n. spicy. (E).", "ee"), "Juniper. bean salad. w.a.t.e.r.m.e.l.o.n. spicy")
    
    testEq(cleanTodoOutput("iScavenging for", "zi"), "Scavenging for")
    
    testEq(cleanTodoOutput("Wakes up at noon (containing the letter 'n') and then goes to the park", "na"), "Wakes up at noon and then goes to the park")
    testEq(cleanTodoOutput("Wakes up at noon (containing the letter 'n') and then goes to the park", "ba"), "Wakes up at noon (containing the letter 'n') and then goes to the park")
    
    testEq(cleanTodoOutput("Wakes up at noon (j for joint) and then goes to the park", "ja"), "Wakes up at noon and then goes to the park")
    
    testEq(cleanTodoOutput("A 'b'urning embers, 'z'ipping like", 'bz'), "A burning embers, zipping like")

    
def removeDate(output):
    
    for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
        pass
        
def removeTrailingPeriod(text):
    # this regex is close paren or two letters, than a period
    return re.sub("((\))|(\w\w))\.$", r"\1", text.strip()).strip() # replace periods at very end, but not itermediate or in stuff like r.o.b.o.t.

    
def cleanTodoOutput(output, letters, debug=False):
    
    output = output.replace("-", " ").strip()
    output = " ".join(splitNoEmpty(output))
    output = removeTrailingPeriod(output)

    letter = f'[{letters.lower()}{letters.upper()}]'
    
    output = re.sub(f"'({letter})'(\w)", r"\1\2", output) # A 'b'urning embers, 'z'ipping like
    output = re.sub(f"\({letter}(,\s*{letter})*\)$", "", output.strip()).strip() # it likes to put (F) or (a, b) at end for some reason, probably because i ask it to include characters
    output = re.sub(f"\({letter}{letter}\)$", "", output.strip()).strip() # it likes to put (AB) at end for similar reasons
    output = re.sub(f"^{letter}{letter}? for", "", output.strip()).strip() # it likes to say "j for ...", remove the first bit
    if not letters.lower() in ["go", 'as', 'at']:
        output = re.sub(f"^{letter}{letter}? ", "", output.strip()).strip() # it likes to say "j ...", remove the first bit
    output = re.sub(f"^{letter}{letter}?:", "", output.strip()).strip() # it likes to say "j:", remove the first bit
    output = re.sub(f"\(containing the letter '{letter}'\)", " ", output.strip()).strip() # it likes to say "j for ...", remove the first bit
    output = re.sub(f"\({letter} for \w+\)", " ", output.strip()).strip() # (j for joint)
    output = " ".join(splitNoEmpty(output))
    
    output = re.sub("^= ", "", output.strip()).strip() # sometimes it likes to make the bullet points pretty
    
    # it puts is for at the front, we don't need that
    output = re.sub("^[Ii]s for", "", output.strip()).strip()
    output = " ".join(splitNoEmpty(output))
    
    # remove the letters if they are in parens like this (AB)
    ll = letters.lower()
    ul = letters.upper()
    output = re.sub(f"(^|\s)\([{ll[0]}{ul[0]}][{ll[1]}{ul[1]}]\)($|\s)", " ", output)
    output = " ".join(splitNoEmpty(output))
    output = re.sub(f"(^|\s)\({letters.upper()}\)($|\s)", " ", output)
    output = " ".join(splitNoEmpty(output))
    
    
    # it likes to do "UVenture" when asked for uv
    # this turns it into Venture
    # or "zApproach nWitch" when asked for z n
    words = splitNoEmpty(output)
    lettersSwapped = letters[1] + letters[0]
    wordsToCheck = [0]
    if len(words) > 1:
        wordsToCheck.append(1)
    for wordI in wordsToCheck:
        word = words[wordI]
        if len(word) >= 3:
            isUpper = [c.upper() == c for c in word]
            # second letter upper, third not
            if isUpper[1] and not isUpper[2]:
                if debug: print("isUpper", isUpper)
                if debug: print("has prefix word")
                # it's a real word, just fix the weird casing
                if word.lower() in WIKI_WORDS:
                    newWord = word[0] + word[1].lower() + word[2:]
                    words[wordI] = newWord
                # they are the letters we asked for
                elif word[0].lower() in letters and word[1:].lower() in WIKI_WORDS:
                    words[wordI] = word[1:]
                    if debug: print("fixing prefix word")
    output = " ".join(words)
    
    
    words = splitNoEmpty(output)
    if letters[0] in ['x', 'q']: # it likes to be cheeky with these and just add it to the front
        for i in range(len(words)):
            w = words[i]
            if len(w) > 2 and w[0].lower() == letters[0] and w[1] == letters[1]:
                if not w in WIKI_WORDS and w[1:] in WIKI_WORDS:
                    words[i] = w[1:]
    output = " ".join(words)
    
    
    
    words = splitNoEmpty(output)
    
    # it likes to abbreviate things
    shortened = {
        'n': 'and',
        'xplore': 'explore',
        'loo(g)': 'log'}
    fixedWords = []
    for w in words:
        if w.lower() in shortened.keys():
            print("replacing", w, shortened[w.lower()])
            fixedWords.append(shortened[w.lower()])
        else:
            fixedWords.append(w)
    output = " ".join(fixedWords)
    
    
    
    
    # this is needed again incase some of the stuff above removed the last word
    output = removeTrailingPeriod(output)    
    
    # uppercase first word
    words = splitNoEmpty(output)
    firstWord = words[0]
    words[0] = firstWord[0].upper() + firstWord[1:]
    output = " ".join(words)
    
    return output.strip()

def upperCaseEveryWord(text):
    words = splitNoEmpty(text)
    return " ".join([w[0].upper() + w[1:] for w in words])

def upperCaseFirstWord(text):
    if len(text.strip()) > 0:
      words = splitNoEmpty(text)
      words[0] = words[0][0].upper() + words[0][1:]
      return " ".join(words)
    else:
      return text

# returns (minVal, maxVal)
# both inclusive
# note, often minVal and maxVal will be the same
def getNumberRange(model, prompt, prefix, topN=1, debug=False):
    res = getNumberAnswer("", prompt, prefix, model, topN=topN, debug=debug)
    prs, numberStr, tokens, fullPrs = res[0]
    parsedResult = parseNumbersOrRange(numberStr)
    numbers = []
    for num, numZeros, numAfterZeros in parsedResult['pieces']:
        numbers.append(multiplyDecimalExact(num, numZeros, numAfterZeros, 1))
    if len(numbers) == 1:
        numbers.append(numbers[0])
    return tuple(sorted(numbers))

def getDuration(model, prompt, prefix, topN=1, debug=False):
    res = getAnswer("", prompt, prefix, model, extractDuration, debug=debug, topN=1)
    prs, pieces, tokens, fullPrs = res[0]
    timeStr = pieces[0][2]
    formattedTimes = "---".join([x[1] for x in pieces])
    return timeStr, formattedTimes
        
def getTodoItemTimeToAccomplish(model, characterPrefix, gender, name, todoPrefix, todoItem, debug=False):
    todoItem = todoItem.replace(".", " ").strip() # remove puncuation
    prompt = characterPrefix + f"\nTask type: {upperCaseFirstWord(todoPrefix)}.\nHow long will it take {name} to {todoItem.lower()}?"
    timeStr, formattedTimes = getDuration(model, prompt, f"That will take about", topN=1, debug=debug)
    return " which will take" + timeStr + " (parsed:" + formattedTimes + ")"


def randomSeed():
    return random.randint(0, 100000000)


def needsFood(model, characterPrefix, name, debug=False):
    prompt = characterPrefix + f"\nDoes {name} ever need to eat? Yes or No."
    return yesVsNo("", prompt, "", model, debug=debug)

def needsAnything(model, characterPrefix, name, debug=False):
    prompt = characterPrefix + f"\nDoes {name} require anything to stay alive? Yes or No."
    return yesVsNo("", prompt, "", model, debug=debug)

def whereLive(model, worldDescription, gender, name, characterDescription):
    pass # TODO



def characterTests(model):
    prompt = """Setting: Earth in modern day
Character: LFAlex
Character species: Algorithm elemental
Character appearance: LFAlex has a kaleidoscope of kinetic, shimmering kryptonite particles swirling around its kobalt blue core
Character personality: Afraid of water and wary of wind, LFAlex's awe-inspiring calculations are its only defense
Character backstory: LFAlex was created by a group of scientists who were attempting to harness the power of lightning, but their fear of water and wary of wind personality made them question their own abilities
Task type: Daily.
How long will it take LFAlex to calculates the optimal path for lightning to strike?"""
    print("if the following gets stuck, rip u, pls fix")
    getAnswer("", prompt, f"That will take about", model, extractDuration, debug=True, topN=1)


# lets think a bit about buildings
# would be nice to give info and let it fill in rest
# for a building, we can split it into rooms
# each room has a type: bedroom, forge room, cooking room, etc
# (technically a room can have multiple purposes but this is a fine simplification)

# for other types of room, for each one we could ask the pr it has a room like that (and maybe the number?), then get the number of non-bedroom rooms and randomly sample?
# can even ask how many rooms on each floor (not only bedrooms, but rooms in general)
#    seems reasonable

class Building(object):
  def __init__(self, buildingType, buildingName, hasMultipleRooms, numPplPerBedroom, numRooms, hasMultipleFloors, numFloors, numRoomsPerFloor):
    floors = []
    if not hasMultipleFloors:
      floors = []
      

# okay so I could do a few things:
# location type, ask if has sub location type?
def subLocationType(model, char, locationType):
    exemptProcessor = exemptStringsProcessor(model, n_context=len(model.context), forbiddenStrings=forbiddenStrings)
    #print("context", model.detokenize(model.context))
    output = promptBulletPoint2(model, debug=debug, rollback=True, logits_processor=exemptProcessor, *args, **kwargs)
            



def getLocationName(model, char, locationType, numNames=1, debug=False, *args, **kwargs):
    prompt = f"{char.characterPrefix}\nTask: {char.name} sometimes visits a {locationType}. This {locationType} does not yet have a name. What might be a good name for this {locationType}?"
    responsePrefix = f"One possible name for a {locationType} is \""
    names = []
    for i in range(numNames):
        names.append(promptNoun(model=model, prompt=prompt, responsePrefix=responsePrefix, forbidden=names))
        print(names[-1])
    return names
    
    


# issue: Sylvia Scythe (see characters/sylviascythe.json) lives in a dark and gritty called The Untamed Underground
# (she's a skeleton), The Untamed Underground will have 10 million ppl

# got "decommissioned" as type? I guess that's fine, didn't see any issues with prompts that got that or any edge cases.

# todo: some locations may have fancy suites, not all rooms should b generated the same
# todo: double check type after getting name (like "bone-chilling" type gives "The Haunted Crypt" but then, prob Crypt should be type)
def getLocationInfo(model, char, seed=None, forceHome=False):
  if seed is None:
      seed = random.randint(0, 100000000)
  hasPermanentResidencePrs = yesVsNo("", char.characterPrefix.strip() + "\n" + f"Task: Does {char.name} have a permanent residence? Yes or No.", "Answer:", model)
  hasPermanentResidence = forceHome or hasPermanentResidencePrs['yes'] >= 0.5
  permanentResidenceType = None
  permanentResidenceName = None
  numPeopleLiveThere = None
  multipleRooms = None
  numRooms = None
  if hasPermanentResidence:
    # this gives weird answers like "bone-chilling" but once we use it to make a name then we can go back and get an actual buildling type
    permanentResidenceTypesorta = promptNoun(model, char.characterPrefix.strip() + "\n" + f"Task: {char.name} has a permanent residence. What kind of building/location is it?", f"Answer: The kind of building/location that {char.name} lives in is a \"").lower()
    permanentResidenceName = promptNoun(model, char.characterPrefix.strip() + "\n" + f"Task: {char.name} has a permanent residence in a {permanentResidenceTypesorta}. What is this location called? {randLetters(3, seed=seed)}", f"Answer: The {permanentResidenceTypesorta} that {char.name} lives in is called \"")
    permanentResidenceType = promptNoun(model, char.characterPrefix.strip() + "\n" + f"Task: {char.name} lives in {permanentResidenceTypesorta} {permanentResidenceName}. What kind of building/location is it?", f"Answer: {upperCaseFirstWord(permanentResidenceTypesorta)} {permanentResidenceName} is a type of building/location known as a \"").lower()
    numPeopleLiveThere = getParsedIntAnswer("", char.characterPrefix.strip() + "\n" + f"Task: {char.name} lives in a {permanentResidenceType} called {permanentResidenceName}. In a typical {permanentResidenceType} similar to {permanentResidenceName}, how many live there?", f"Answer: The number that live in a typical {permanentResidenceType} similar to {permanentResidenceName} is about", model, topN=2)
    # "typical" is needed bc otherwise "that info is not provided" hogs all the pr mass, meaning the numbers are low pr 
    multipleRoomsPrs = yesVsNo("", char.characterPrefix.strip() + "\n" + f"Task: {char.name} lives in a {permanentResidenceType} called {permanentResidenceName}. Does {permanentResidenceName} have more than one room to live in? Yes or No.", f"Answer:", model)
    hasMultipleRooms = multipleRoomsPrs['yes'] >= 0.5
    numPplPerBedroom = 1
    numRooms = 1
    if hasMultipleRooms:
      numRooms = getParsedIntAnswer("", char.characterPrefix.strip() + "\n" + f"Task: {char.name} lives in a {permanentResidenceType} called {permanentResidenceName}. Within {permanentResidenceName} there are multiple bedrooms. How many bedrooms would a typical {permanentResidenceType} like {permanentResidenceName} have?", f"Answer: The number of bedrooms in a typical location similar to {permanentResidenceType} {permanentResidenceName} is about" , model, topN=2)
      # note: don't give it the number of bedrooms or it gets confused and will put that for how many ppl in each bedroom as well
      numPplPerBedroom = getParsedIntRangeAnswer("", char.characterPrefix.strip() + "\n" + f"Task: {char.name} lives in a {permanentResidenceType} called {permanentResidenceName}. Within {permanentResidenceName} there are multiple bedrooms. In a typical {permanentResidenceType} like {permanentResidenceName}, how many stay in each bedroom?", f"Answer: In a typical location similar to {permanentResidenceType} {permanentResidenceName}, the number that stay in each bedroom is about" , model, topN=2)
    multipleFloorsPrs = yesVsNo("", char.characterPrefix.strip() + "\n" + f"Task: {char.name} lives in a {permanentResidenceType} called {permanentResidenceName}. Would a typical building like {permanentResidenceName} have multiple floors? Yes or No.", f"Answer:", model)
    hasMultipleFloors = multipleFloorsPrs['yes'] >= 0.5
    numFloors = 1
    numRoomsPerFloor = 1
    if hasMultipleFloors:
      numFloors = getParsedIntAnswer("", char.characterPrefix.strip() + "\n" + f"Task: {char.name} lives in a {permanentResidenceType} called {permanentResidenceName}. {permanentResidenceName} has multiple floors. How many floors would a typical {permanentResidenceType} like {permanentResidenceName} have?", f"Answer: The number of floors of a typical {permanentResidenceType} like {permanentResidenceName} is about", model, topN=2)
      # don't give num floors or that biases the output too much
      numRoomsPerFloor = getParsedIntRangeAnswer("", char.characterPrefix.strip() + "\n" + f"Task: {char.name} lives in a {permanentResidenceType} called {permanentResidenceName}. Within {permanentResidenceName} there are multiple floors to live in. On each floor, how many bedrooms would a typical {permanentResidenceType} like {permanentResidenceName} have?", f"Answer: The number of bedrooms per floor of a typical {permanentResidenceType} like {permanentResidenceName} is about", model, topN=2)
    hasOtherRoomTypes = yesVsNo("", char.characterPrefix.strip() + "\n" + f"Task: {char.name} lives in a {permanentResidenceType} called {permanentResidenceName}. Does a typical building like {permanentResidenceName} have rooms that are not bedrooms? Yes or No.", "Answer:", model)
    otherRoomTypes = []
    if hasOtherRoomTypes:
      otherTypes = promptMultiBulletPoint(model, char.characterPrefix.strip() + "\n" + f"Task: {char.name} lives in a {permanentResidenceType} called {permanentResidenceName}. {permanentResidenceName} has rooms that are not bedrooms. What kind of rooms are these?", f"The {permanentResidenceType} {permanentResidenceName} has the following types of non-bedroom rooms:\n-", postProcess=trimSingleItem, num=10, debug=False, stop=stopSingleItem)

  
  for k,v in locals().items():
    print(k + ": " + str(v) + "\n\n")
  return hasPermanentResidencePrs, hasPermanentResidence, permanentResidenceType, permanentResidenceName, numPeopleLiveThere, multipleRooms, numRooms
# todo: figure out if daily or one time task
# for example: Practice sword fighting daily at the training arena
# todo: remove duplicate tasks



# ask for morning, afternoon, evening?
# types of tasks they might do? (as a list)
# then for each type, when they might do it (morning, afternoon, evening)
# note it is common to put characterInfo = "{name} is a (whatever species they are)", this helps adherence to species
class Character(object):
    def __init__(self, model, worldDescription, gender, species='human', appearance=None, characterInfo=None, name=None, personality=None, backstory=None, seed=None, debug=False, **kwargs):
        self.worldDescription = worldDescription
        self.gender = gender
        species = upperCaseFirstWord(species)
        self.species = species
        random.seed(seed)
        self.name = name
        
        kwargsStr = ""
        if len(kwargs) > 0:
            kwargsStr = "\n" + "\n".join([("Character " + str(k).lower() + ": " + str(upperCaseFirstWord(removeTrailingPeriod(v)))) for (k,v) in kwargs.items()])
        if name is None:
            self.name = getName(model, worldDescription, gender, species=species, characterInfo=characterInfo, seed=randomSeed(), debug=debug, kwargsStr=kwargsStr)
        
        if not characterInfo is None:
            characterInfo = characterInfo.replace("{name}", self.name)
        
        self.appearance = appearance
        if appearance is None:
           self.appearance = getCharacterAppearance(model, worldDescription, gender, name=self.name, species=species, characterInfo=characterInfo, seed=randomSeed(), debug=debug, kwargsStr=kwargsStr) 
        
        self.personality = personality
        if personality is None:
            self.personality = getCharacterPersonality(model, worldDescription, gender, name=self.name, appearance=self.appearance, species=species, characterInfo=characterInfo, seed=randomSeed(), debug=debug, kwargsStr=kwargsStr)
       
        self.backstory = backstory
        if backstory is None:
            self.backstory = getCharacterBackstory(model, worldDescription, gender, name=self.name, species=species, appearance=self.appearance, personality=self.personality, characterInfo=characterInfo, seed=randomSeed(), debug=debug, kwargsStr=kwargsStr)
       
        self.characterPrefix = f"Setting: {removeTrailingPeriod(worldDescription)}\nCharacter: {self.name}{kwargsStr}\nCharacter species: {upperCaseFirstWord(species)}\nCharacter appearance: {removeTrailingPeriod(self.appearance)}\nCharacter personality: {removeTrailingPeriod(self.personality)}\nCharacter backstory: {removeTrailingPeriod(self.backstory)}"
        
        return
        self.todoPrefixes = ['daily', 'weekly', 'monthly', 'long term']
        self.todos = {}
        for prefix in self.todoPrefixes:
            todo = [getTodoItem(model, self.characterPrefix, gender, self.name, prefix=prefix, seed=randomSeed(), debug=debug) for _ in range(0)]
            durations = [getTodoItemTimeToAccomplish(model, self.characterPrefix, gender, self.name, prefix, item, debug=debug) for item in todo]
            self.todos[prefix] = [(x + str(i)) for (x,i) in zip(todo, durations)]
        
        self.locations = []
        
        isPlantPrompt = self.characterPrefix + f"\nIs {self.name} a plant? Yes or No."
        self.isPlantPrs = yesVsNo("", isPlantPrompt, "", model, debug=debug)
        self.isPlant = self.isPlantPrs['yes'] >= 0.5
        
        needsFoodPrompt = self.characterPrefix + f"\nDoes {self.name} ever need to eat? Yes or No."
        self.needsFoodPrs = yesVsNo("", needsFoodPrompt, "", model, debug=debug)
        self.needsFood = self.needsFoodPrs['yes'] >= 0.5
        
        needsElectricityPrompt = self.characterPrefix + f"\nDoes {self.name} need electricity and/or need to charge in order to continue functioning? Yes or No."
        self.needsElectricityPrs = yesVsNo("", needsElectricityPrompt, "", model, debug=debug)
        self.needsElectricity = self.needsElectricityPrs['yes'] >= 0.5
        

        whatTasksPrompt = self.characterPrefix + f"\nWhat types of tasks does {self.name} have on their todo list? Keep your responses concise, one to five words."
        
        tasks = promptMultiBulletPoint(model, whatTasksPrompt, f"{self.name} has the following types of tasks on {pronoun[self.gender].lower()} todo list:\n-", postProcess=trimSingleItem, num=10, debug=debug, stop=stopSingleItem)
        
        print(tasks)


        whatNeedsPrompt = self.characterPrefix + f"\nWhat physical materials/objects/energy/etc. does {self.name} need in order to continue existing/stay alive? Keep your responses concise, one to five words."
        
        needs = promptMultiBulletPoint(model, whatNeedsPrompt, f"{self.name} requires obtaining the following types of things to continue existing/stay alive:\n-", postProcess=trimSingleItem, num=10, debug=debug, stop=stopSingleItem)
        self.whatNeeds = []
        for need in needs:
            def lowerFirstChar(s):
                return s[0].lower() + s[1:]
            
            fillNeedDailyPrompt = self.characterPrefix + f'\nQuestion: If {self.name} was a real character, would {pronoun2[gender].lower()} typically fill their need for {lowerFirstChar(need)} daily? Yes or No.'
            fillNeedDailyPrs = yesVsNo("", fillNeedDailyPrompt, "Answer:", model, debug=debug)
            fillNeedDaily = fillNeedDailyPrs['yes'] >= 0.5
            timesPerDay = 0
            if fillNeedDaily:
                howOftenGetPrompt = self.characterPrefix + f'\nQuestion: If {self.name} was a real character, how many times a day would they take action to meet their need for {lowerFirstChar(need)}?'
                howOftenGetPrefix = f'Answer: The average number of times per day that {self.name} does something to fill their need for {lowerFirstChar(need)} is'
                timesPerDay = getNumberRange(model, howOftenGetPrompt, howOftenGetPrefix, topN=1, debug=debug)
            #else:
            #    howOftenGetPrompt = self.characterPrefix + f'\nQuestion: If {self.name} was a real character, how often would they take action to meet their need for {lowerFirstChar(need)}?'
            #    howOftenGetPrefix = f'{self.name} typically does something to meet their need for {lowerFirstChar(need)} every '
            #    timeStr, formattedTimes = getDuration(model, howOftenGetPrompt, howOftenGetPrefix, topN=1, debug=debug)
            
            
            # The "if was real" is necessary because otherwise it says "no, they are a fictional character so don't require [stuff like air or food and water]"
            isNecessaryPrompt = self.characterPrefix + f"\nQuestion: If {self.name} was a real character, would {pronoun2[gender].lower()} need {lowerFirstChar(need)} in order to stay alive? Yes or No."
            isNecessaryPrs = yesVsNo("", isNecessaryPrompt, "Answer:", model, debug=debug)
            isNecessary = isNecessaryPrs['yes'] >= 0.5
            
            self.whatNeeds.append((need, "fill need daily", fillNeedDaily, fillNeedDailyPrs, "times per day", timesPerDay, "necessary", isNecessary, isNecessaryPrs))
        
        
        
        # most plants will get no because they don't "eat", but still might require water or good soil conditions
        if not self.needsFood:
            self.needsAnythingPrs = needsAnything(model, self.characterPrefix, name=self.name, debug=debug)
            self.needsAnything = self.needsAnythingPrs['yes'] >= 0.5
        else:
            self.needsAnythingPrs = {"yes": 1.0, "no": 0.0}
            self.needsAnything = True
        
    def save(self, fileName):
      with codecs.open(fileName, "w", 'utf-8') as f:
        f.write(jsonpickle.dumps(self))
    
    def __repr__(self):
        res = "\n\n".join([name + ": " + str(value) for (name, value) in vars(self).items()])
        return res
        todosStr = "\n".join([(prefix + " todos:\n-" + "\n-".join(todos)) for (prefix, todos) in self.todos.items()])
        locationsStr = "\n".join([("locations:\n-" + "\n-".join(locations)) for (locations) in [self.locations]])
        return f"Name: {self.name}\nAppearance: {self.appearance}\nPersonality: {self.personality}\nBackstory: {self.backstory}\n{todosStr}\n{locationsStr}\nNeeds: {self.whatNeeds}\nIs a plant or fungus? {str(self.isPlant)}({str(self.isPlantPrs)})\nNeeds food? {str(self.needsFood)}({str(self.needsFoodPrs)})\nNeeds electricity? {str(self.needsElectricity)}({str(self.needsElectricityPrs)})\nNeeds anything for sustinence? {str(self.needsAnything)}({str(self.needsAnythingPrs)})"
  

def loadChar(fileName):
  with codecs.open(fileName, "r", "utf-8") as f:
    text = f.read()
    return jsonpickle.loads(text)

def doFrequencyCounts(model, generateFun, seed=None):
    if seed is None:
        seed = random.randint(0, 100000000)
    totals = defaultdict(lambda: 0)
    wordTotals = []
    t = 0
    print("using seed", seed)
    try:
        while True:
            output = generateFun(seed+t)
            print(output)
            totals[output] += 1
            words = output.split()
            while len(words) > len(wordTotals):
                wordTotals.append(defaultdict(lambda: 0))
            for i, word in enumerate(words):
                wordTotals[i][word] += 1
            t += 1
    except InterruptedException:
        print("-----totals-----")
        res = sorted(list(totals.items()), key=lambda x: x[1])
        for w,c in res: print(w + ":", c)
        
        print()
        print("-----word totals-----")
        for i in range(len(wordTotals)):
            print("--word--", i)
            res = sorted(list(wordTotals[i].items()), key=lambda x: x[1])
            for w,c in res: print(w + ":", c)
            print()
            
# idea:
# make list of todos
# for each todo:
#  - determine if in multiple locations or one
#  - generate the type of locations needed
#  - generate the order of type of locations traversed
#  - determine how long it'll take to finish
#  - determine how long to work on it each day?
# for each type of location:
#  - generate if it can be within an existing location or if we need a new one (city, town, etc.)
#    - for containment, ask how many of that location occurs 
#  - generate a name (using the "must contain letters" trick for diversity)
# 
def setAllSeed(seed):
  #random.seed(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  
def npSoftmax(vals):
    e_x = np.exp(vals-np.max(vals)) # log exp sum trick for numerical stability
    return e_x/np.sum(e_x)
  
def softmax(vals):
    maxVal = max(vals)
    e_x = [math.exp(v - maxVal) for v in vals] # log exp sum trick for numerical stability
    total = sum(e_x)
    return [(x/total) for x in e_x]

def sampleFromSet(instruction, prompt, responsePrefix, model, debug=False, explain=False):
    model.reset() # it's stateful so without this the previous evals will impact us
    modelInput = toPromptFancy(instruction, prompt, responsePrefix)
    model.eval(model.tokenize(modelInput))
    if debug:
        print(modelInput, end="", flush=True)
    yesStrings = ['Yes', ' Yes', 'yes', ' yes'] # space in front is seperate token, we want sum over all possibles
    noStrings = ['No', ' No', 'no', ' no']
    yesTokens = [model.tokenize(s)[1] for s in yesStrings] # first is start token
    noTokens = [model.tokenize(s)[1] for s in noStrings] # first is start token
    logits = model.logits
    yesLogits = [logits[t] for t in yesTokens]
    noLogits = [logits[t] for t in noTokens]
    resultPrs = softmax(yesLogits + noLogits)
    yesPr = sum(resultPrs[:len(yesLogits)])
    noPr = sum(resultPrs[len(yesLogits):])
    if debug: print(list(zip(yesStrings+noStrings, resultPrs)))
    if debug: print("Y:", yesPr, "N:", noPr)
    
    if explain:
      model.generatep(instruction, prompt, responsePrefix + "Yes,")
      model.generatep(instruction, prompt, responsePrefix + "Yes,")
      model.generatep(instruction, prompt, responsePrefix + "No,")
      model.generatep(instruction, prompt, responsePrefix + "No,")
    
    return {"yes": yesPr, "no": noPr}



def yesVsNo2(model, debug=False, explain=False):
    if debug:
        print(model.detokenize(model.context), end="", flush=True)
    yesStrings = ['Yes', ' Yes', 'yes', ' yes'] # space in front is seperate token, we want sum over all possibles
    noStrings = ['No', ' No', 'no', ' no']
    yesTokens = [model.tokenize(s)[1] for s in yesStrings] # first is start token
    noTokens = [model.tokenize(s)[1] for s in noStrings] # first is start token
    logits = model.logits
    yesLogits = [logits[t] for t in yesTokens]
    noLogits = [logits[t] for t in noTokens]
    resultPrs = softmax(yesLogits + noLogits)
    yesPr = sum(resultPrs[:len(yesLogits)])
    noPr = sum(resultPrs[len(yesLogits):])
    if debug: print(list(zip(yesStrings+noStrings, resultPrs)))
    if debug: print("Y:", yesPr, "N:", noPr)
    
    if explain:
      print("explaining")
      curLogits = model.logits
      y = model.tokenize("Yes,")[1:]
      model.eval(y)
      print("Yes," + model.generate2(rollback=True, stop=['\n'], numTokens=100)['str'])
      print("Yes," + model.generate2(rollback=True, stop=['\n'], numTokens=100)['str'])
      for t in y: model.popcontext()
      n = model.tokenize("No,")[1:]
      print("No," + model.generate2(rollback=True, stop=['\n'], numTokens=100)['str'])
      print("No," + model.generate2(rollback=True, stop=['\n'], numTokens=100)['str'])
      for t in n: model.popcontext()
      model.logits = curLogits
      
    return {"yes": yesPr, "no": noPr}


def yesVsNo(instruction, prompt, responsePrefix, model, wordSet, debug=False, explain=False):
    model.reset() # it's stateful so without this the previous evals will impact us
    modelInput = toPromptFancy(instruction, prompt, responsePrefix)
    model.eval(model.tokenize(modelInput))
    if debug:
        print(modelInput, end="", flush=True)
    yesStrings = ['Yes', ' Yes', 'yes', ' yes'] # space in front is seperate token, we want sum over all possibles
    noStrings = ['No', ' No', 'no', ' no']
    yesTokens = [model.tokenize(s)[1] for s in yesStrings] # first is start token
    noTokens = [model.tokenize(s)[1] for s in noStrings] # first is start token
    logits = model.logits
    yesLogits = [logits[t] for t in yesTokens]
    noLogits = [logits[t] for t in noTokens]
    resultPrs = softmax(yesLogits + noLogits)
    yesPr = sum(resultPrs[:len(yesLogits)])
    noPr = sum(resultPrs[len(yesLogits):])
    if debug: print(list(zip(yesStrings+noStrings, resultPrs)))
    if debug: print("Y:", yesPr, "N:", noPr)
    
    if explain:
      model.generatep(instruction, prompt, responsePrefix + "Yes,")
      model.generatep(instruction, prompt, responsePrefix + "Yes,")
      model.generatep(instruction, prompt, responsePrefix + "No,")
      model.generatep(instruction, prompt, responsePrefix + "No,")
    
    return {"yes": yesPr, "no": noPr}



def argmax(arr):
    return np.argmax(arr)
    '''
    maxVal = arr[0]
    maxInd = 0
    for i in range(len(arr)):
        if arr[i] > maxVal:
            maxVal = arr[i]
            maxInd = i
    return maxInd
    '''
    
def testRollback(instruction, prompt, responsePrefix, model, nTokens):
    model.reset() # it's stateful so without this the previous evals will impact us
    model.eval(model.tokenize(toPromptFancy(instruction, prompt, responsePrefix)))
    logits = model.logits
    import numpy as np
    topTokens = np.argsort(-np.array(logits))
    
    model.eval([topTokens[0]])
    resPrs = model.logits[0], model.logits[3]
    model.popcontext()
    
    model.eval([topTokens[0]])
    resPrs2 = model.logits[0], model.logits[3]
    model.popcontext()
    print(resPrs)
    print(resPrs2)
    
    print("attempt to decode n tokens using", model.detokenize([topTokens[0]]))
    model.eval([topTokens[0]])
    print(model.detokenize([topTokens[0]]), end="", flush=True)
    parseTokens(model, nTokens)
    print("\n done, rolling back\n")
    for i in range(nTokens+1):
        model.popcontext()
        
    print("attempt to decode n tokens using", model.detokenize([topTokens[0]]))
    model.eval([topTokens[0]])
    print(model.detokenize([topTokens[0]]), end="", flush=True)
    parseTokens(model, nTokens)
    print("\n done, rolling back\n")
    for i in range(nTokens+1):
        model.popcontext()
    
    
    print("attempt to decode n tokens using", model.detokenize([topTokens[1]]))
    model.eval([topTokens[1]])
    print(model.detokenize([topTokens[1]]), end="", flush=True)
    parseTokens(model, nTokens)
    print("\n done, rolling back\n")
    for i in range(nTokens+1):
        model.popcontext()
    
    
   
    
def testSingleToken(model):
    logits = model.logits
    maxLogit = argmax(logits)
    logitToken = model.detokenize([maxLogit])
    print(logitToken)
    model.eval([maxLogit])

def rawParse(instruction, prompt, responsePrefix, model, nTokens, reset=True):
    if reset: model.reset()
    model.eval(model.tokenizer(toPromptFancy(instruction, prompt, responsePrefix))['input_ids'])
    parseNTokensBetter(model, nTokens)
    
def parseTokens(model, nTokens):
    for i in range(nTokens):
        logits = model.logits
        maxLogit = argmax(logits)
        logitToken = model.detokenize([maxLogit])
        print(logitToken, end="", flush=True)
        model.eval([maxLogit])
    
# howto argmax parsing, one token at a time, using ctransformers
def testParse(instruction, prompt, responsePrefix, model, nTokens, reset=True):
    model.reset() # it's stateful so without this the previous evals will impact us
    model.eval(model.tokenize(toPromptFancy(instruction, prompt, responsePrefix)))
    
    
class PrecomputedParseInfo(object):
    def __init__(self, model):
        # populate the logits
        self.tokens = []
        for i in range(len(model.tokenizer.vocab)):
            self.tokens.append(model.idToToken(i))
        
        hasNumbersRegex = re.compile(r'[0123456789]')
        self.numberTokens = []
        self.numberTokenSet = set()
        for i, t in enumerate(self.tokens):
            if hasNumbersRegex.search(t) and not (t[0] == "<" and t[-1] == '>'):
                self.numberTokens.append((i, t))
                self.numberTokenSet.add(i)
        self.commaToken = model.tokenToId(",") # all the other stuff with comma aren't useful for number parsing
        
        self.numberParseTokens = [[t] for t in self.numberTokenSet] + [[self.commaToken]]
        numberWords = NUMBER_WORDS + ['and']
        
        
        startToken, spaceAndDash = model.tokenizer("-")['input_ids']
        for numberWord in numberWords:
            for w in [numberWord, numberWord[0].upper() + numberWord[1:]]: # upper and lower case
                self.numberParseTokens.append(model.tokenizer(w)['input_ids'][1:]) # exclude the <s>, this has a space in front
                def filterStartTokenAndDashAndSpace(tokens):
                    return [x for x in tokens if not x in [startToken,spaceAndDash]] # 1 is <s>, 448 is " -"
                self.numberParseTokens.append(filterStartTokenAndDashAndSpace(model.tokenizer('-' + w)['input_ids'])) # a version without the space, needed for stuff like twenty-two or four-thousand
        
        three = model.tokenToId("3")
        start = model.tokenToId("<s>")
        self.spaceToken = list(set([t for t in model.tokenizer("3 3")['input_ids'] if not t in [three, start]]))[0]
        self.numberParseTokens.append([self.spaceToken])
        self.dashToken = model.tokenToId('-')
        self.numberParseTokens.append([self.dashToken])
        self.periodToken = model.tokenToId(".")
        self.numberParseTokens.append([self.periodToken])
        #for toks in self.numberParseTokens:
        #    print("|".join([model.idToToken(t) for t in toks]))
        
        self.numberParseTokensSet = set()
        for tokens in self.numberParseTokens:
            self.numberParseTokensSet.update(tokens)
        
        self.durationTokens = []
        durationWords = ['millennium', 'century', 'centurie', 'decade', 'year', 'quarter', 'season', 'month', 'week', 'day', 'hour', 'minute', 'millisecond', 'microsecond']
        
        def yrsToMilliseconds(yrs):
            return yrs*daysToMilliseconds(365) # ~365 day per year, 
        
        def monthsToMilliseconds(months):
            return months*daysToMilliseconds(30) # ~30 days in a month
            
        def daysToMilliseconds(days):
            return days*hoursToMilliseconds(24) # 24 hrs per day
        
        def hoursToMilliseconds(hours):
            return hours*60*60*1000 # 60 minutes per hour, 60 seconds per minute, 1000 milliseconds per second
        
        self.durationToMillis = {
            'millennium': yrsToMilliseconds(1000),
            'century': yrsToMilliseconds(100),
            'centurie': yrsToMilliseconds(100),
            'decade': yrsToMilliseconds(10),
            'year': yrsToMilliseconds(1),
            'quarter': monthsToMilliseconds(3),
            'season': monthsToMilliseconds(3),
            'month': monthsToMilliseconds(1),
            'week': daysToMilliseconds(7),
            'day': daysToMilliseconds(1),
            'hour': hoursToMilliseconds(1),
            'minute': 60*1000,
            'millisecond': 1,
            'microsecond': 0
        }
        
        # add plurals
        for k,v in list(self.durationToMillis.items()):
            self.durationToMillis[k + 's'] = v
        
        for d in durationWords:
            for dw in [d, d+'s']: #plural
                for dwi in [dw, dw[0].upper() + dw[1:]]: # uppercase and lowercase
                    
                    self.durationTokens.append(model.tokenizer(dwi)['input_ids'][1:]) # exclude the <s>, this has a space in front
                    def filterStartTokenAndDashAndSpace(tokens):
                        return [x for x in tokens if not x in [startToken,spaceAndDash]] # 1 is <s>, 448 is " -"
                    self.durationTokens.append(filterStartTokenAndDashAndSpace(model.tokenizer('-' + dwi)['input_ids'])) # a version without the space

                    
        
        # note to self: while it is tempting to forbid stuff like
        # fourty-fourty (and only allow fourty-four), note that
        # fourty-fourty five (as in, a range of fourty to fourty five)
        # is valid.
            
            
            

global parseInfo
parseInfo = None

def getParseInfo(model):
    global parseInfo
    if parseInfo is None:
        parseInfo = PrecomputedParseInfo(model)
    return parseInfo
    
    
NONTOKEN = 'nontoken'
    
    
def getPr(model, tokens):
    allPrs = []
    for t in tokens:
        prs = npSoftmax(model.logits)
        allPrs.append(prs[t])
        model.eval([t])
    return accumulatePrs(allPrs), allPrs
        
        
def parseNTokensBetter(model, nTokens):
    for t in range(nTokens):
        t = model.llm.sample()
        print(t, model.idToToken(t))
        model.eval([t])
        
    
def parseTokenNonExclusive(model, logits=None, tokens=None):
    if logits is None:
        logits = model.logits
    parseInfo = getParseInfo(model)
    prs = npSoftmax(np.array(logits))
    digitPrs = defaultdict(lambda: 0)
    for toks in tokens:
        digitPrs[tuple(toks)] += prs[toks[0]]
    resultPrs = [(list(toks), pr) for (toks, pr) in digitPrs.items()]
    nonDigitPr = 1.0-sum([pr for (toks, pr) in resultPrs])
    resultPrs.append((NONTOKEN, nonDigitPr))
    resultPrs.sort(key=lambda x: -x[1])
    return resultPrs

def getParsedIntRangeAnswer(instruction, prompt, responsePrefix, model, topN, debug=False):
  #print()
  #generate(instruction, prompt, responsePrefix, model)
  #print()
  numbers = getNumberAnswer(instruction, prompt, responsePrefix, model, topN, debug=True)
  # this logic is so if first output is 10, and second is 10-15 (which will always b lower pr by definition),
  #  we will prefer the range (because the range and second both start with 10)
  if len(numbers) > 1:
    # we only care about stuff to left of decimal point
    num1 = parseNumbersOrRange(numbers[0][1])
    num2 = parseNumbersOrRange(numbers[1][1])
    if num1['type'] == 'value' and num2['type'] == 'range':
      num1Num, _, _ = num1['pieces'][0]
      (num2Start, _, _), (num2End, _, _) = num2['pieces']
      print("\nhave num1 of", num1Num, "and num2 range of", num2Start, num2End)
      if num2Start == num1Num:
        print("matched nums, returning second")
        return parseIntRangeFromText(numbers[1][1]), numbers
      else:
        print("did not match")
    # if both are single values, do a weighted range according to their prs
    if num1['type'] == 'value' and num2['type'] == 'value':
      num1Num, _, _ = num1['pieces'][0]
      num2Num, _, _ = num2['pieces'][0]
      num1Pr = numbers[0][0]
      num2Pr = numbers[1][0]
      print("both ints, using weighted range")
      return WeightedRange(minVal=num1Num, minPr=num1Pr, maxVal=num2Num, maxPr=num2Pr), numbers

  return parseIntRangeFromText(numbers[0][1]), numbers

def getParsedIntAnswer(instruction, prompt, responsePrefix, model, topN, debug=False):
  intRange, rawData = getParsedIntRangeAnswer(instruction=instruction, prompt=prompt, responsePrefix=responsePrefix, model=model, topN=topN, debug=debug)
  return round(intRange.sample()), rawData

def getNumberAnswer(instruction, prompt, responsePrefix, model, topN, debug=False):
    model.reset()
    prompt = toPromptFancy(instruction, prompt, responsePrefix)
    if debug or model.fullDebug: print(prompt, end="", flush=True)
    model.eval(model.tokenize(prompt))
    return parseNumber(model, topN)

def getAnswerp(instruction, prompt, responsePrefix, model, func, *args, **kwargs):
    model.reset()
    model.eval(model.tokenize(toPromptFancy(instruction, prompt, responsePrefix)))
    res = func(model, *args, **kwargs)
    for k in res:
        print(k)

def getAnswer(instruction, prompt, responsePrefix, model, func, debug=False, *args, **kwargs):
    model.reset()
    model.eval(model.tokenize(toPromptFancy(instruction, prompt, responsePrefix)))
    if debug:
        print(toPromptFancy(instruction, prompt, responsePrefix))
    return func(model, *args, **kwargs)
    
def accumulatePrs(prs):
    # the total pr is product of all prs
    # however, this unfairly prioritizes short things
    # so we do productOfPrs^(1.0/numTokens)
    # there's also many things that are > 0.98
    # throw those out since they don't matter
    prsFiltered = [p for p in prs if p < 0.98]
    if len(prs) == 0: prsFiltered = prs # incase they are all 0.98 or higher
    return np.product(prsFiltered)
    #return np.product(np.power(prs, 1.0/len(prs))) # has issues encouraging stuff like 40000000000
    #return np.product(prs) doesn't work because something like "four" will always be preferred over fourteen, see for example for x in world.getAnswer("", "How many people are in a town with fourteen people?", "There are", model, world.parseNumber, 8): print(x)


def generate(instruction, prompt, responsePrefix, model, debug=False):
    modelInput = toPromptFancy(instruction, prompt, responsePrefix)
    if debug and not model.fullDebug:
        print(modelInput, end="", flush=True)
    for isDone, t in model.generate(modelInput):
        if not t is None and (debug and not model.fullDebug): print(t, end="", flush=True)

# while this works, I find it is better to just ask for the minimum and maximum (say, population of X)
def parseNumberRange(model, topN1, expandN1, topN2):
    results1 = parseNumber(model, topN1)
    results = SortedList()
    for pr, digits, tokens, prs in results1[:expandN1]:
        model.eval(tokens + [model.tokenToId("-")])
        results2 = parseNumber(model, topN2)
        
        for pr2, digits2, tokens2, prs2 in results2:
            totalPr = accumulatePrs(prs + prs2)
            #print("1", pr, digits, tokens, prs)
            #print("2", pr2, digits2, tokens2, prs2)
            resToAdd = (-totalPr, digits, digits2, prs+prs2)
            results.add(resToAdd)
            #print(resToAdd)
            
        # rollback
        for _ in range(len(tokens)+1):
            model.popcontext()
    
    print("Final results:")
    
    # needed negative for sorting
    results = [(-totalPr, digits, digits2, prs) for (totalPr, digits, digits2, prs) in results]
    
    for totalPr, digits1, digits2, prs in results:
        print(totalPr, np.product(prs), np.product([x for x in prs if x < 0.98]), accumulatePrs([x for x in prs if x < 0.98]), digits1 + "-" + digits2, prs)


def timeIt(func, *args, **kwargs):
    startTime = timeInMillis()
    res = func(*args, **kwargs)
    print("Took:", ((timeInMillis() - startTime)/1000.0), "seconds")
    return res




def greedyDecodeFromTokens(model, tokenList, forceFirst=True, rollback=True, debug=False):
    numTokensParsed = 0
    resultTokenLists = []
    resultPrLists = []
    
    thisRunResultTokens = []
    thisRunResultPrs = []
    while True:
        logits = model.logits
        prs = npSoftmax(logits)
        def firstOrItem(stuff): # support some of the tokens being lists of tokens (like how Thousand is mulitple tokens)
            if type(stuff) is list:
                return stuff[0]
            else:
                return stuff
        tokenPrs = [prs[firstOrItem(tokens)] for tokens in tokenList]
        # something else higher pr than tokens, bail
        if np.max(prs) > np.max(tokenPrs) and not forceFirst:
            if debug: print("failed w context", model.tokenizer.decode(thisRunResultTokens), "better next choice is", model.idToToken(np.argmax(prs)), "with pr", np.max(prs))
            break
        forceFirst = False
        # if there are ties, do both of them (like if we have eighteen and eight)
        bestInList = np.argsort(-np.array(tokenPrs))
        winners = [tokenList[bestInList[0]]]
        bestPr = tokenPrs[bestInList[0]]
        for i in range(1, len(tokenPrs)):
            winnerI = bestInList[i]
            pr = tokenPrs[winnerI]
            if pr == bestPr:
                winners.append(tokenList[winnerI])
            else:
                break
        
        bonusResultTokens = []
        bonusResultPrs = []
        curContext = model.tokenizer.decode(thisRunResultTokens)
        if debug: print("\ncontext:", curContext)
        if "." in curContext and len(curContext.split(".")[-1]) > 5:
            print("bailing precise decimal points", curContext) # stuff like 0.00000000
            break
        #print("winners:")
        #for w in winners:
        #    if not type(w) is list:
        #        w = [w]
        #    #print(model.tokenizer.decode(w), w)
        for i, bestToken in enumerate(winners):
            if not type(bestToken) is list:
                bestToken = [bestToken]
            # process all the tokens we chose
            #\print("\nstart")
            #print("context", model.tokenizer.decode(thisRunResultTokens))
            curPrs = prs
            for t in bestToken:
                thisRunResultTokens.append(t)
                thisRunResultPrs.append(curPrs[t])
                model.eval([t])
                curPrs = npSoftmax(model.logits)
                numTokensParsed += 1
                #print("-processed", model.idToToken(t))
                #print("-context", model.tokenizer.decode(thisRunResultTokens))
                #print("-prs", thisRunResultPrs)
            if i != len(winners)-1: # we need to use recursion to finish up others
                # because ties shouldn't happen very often this shouldn't hit recursion limit
                #print("--recursing:--")
                bonusTokenLists, bonusPrLists = greedyDecodeFromTokens(model, tokenList, forceFirst=False, rollback=True)
                for bonusTokens, bonusPrs in zip(bonusTokenLists, bonusPrLists):
                    
                    bonusTokens = thisRunResultTokens + bonusTokens
                    bonusPrs = thisRunResultPrs + bonusPrs
                    
                    #print("got bonus")
                    #print(bonusTokens, bonusPrs)
                    #print("bonus str is", model.tokenizer.decode(bonusTokens))
                    
                    resultTokenLists.append(bonusTokens)
                    resultPrLists.append(bonusPrs)
                
                # roll it back
                for t in bestToken:
                    numTokensParsed -= 1
                    model.popcontext()
                    thisRunResultTokens.pop()
                    thisRunResultPrs.pop()
                
    resultTokenLists.append(thisRunResultTokens)
    resultPrLists.append(thisRunResultPrs)
    # rollback
    if rollback:
        for _ in range(numTokensParsed):
            model.popcontext()
    
    return resultTokenLists, resultPrLists    


def prettyPrintDuration(milliseconds):
    return str(timedelta(milliseconds=milliseconds))
    

# Given 4.065 times 2000
# num = 4
# numZeros = 1
# numAfterZeros = 65
# numMultiply = 2000
# note this rounds away anything after a decimal and always returns an integer
def multiplyDecimalExact(num, numZeros, numAfterZeros, numMultiply):
    # stuff left of period
    result = numMultiply * num
    # stuff right of period
    # 0.3*numMultiply is same as 3*numMultiply//10
    # 0.03*numMultiply is same as 3*numMultiply//100
    resultLeftOfDigit = (numAfterZeros*numMultiply)//(10**(numZeros+1))
    return result+resultLeftOfDigit
    
    
def extractDuration(model, topN, debug=False):
    results = []
    parseInfo = getParseInfo(model)
    for totalP, (numStr, durationStr), tokens, pArr in parseDuration(model, topN):
        parsedNum = parseNumbersOrRange(numStr)
        durationToks = splitNoEmpty(durationStr)
        if len(durationToks) > 1:
            print("warning, got:", durationToks, "for duration, should only have one token, only looking at first token")
        durationTok = durationToks[0].strip().lower()
        resultPieces = []
        durationInMilliseconds = parseInfo.durationToMillis[durationTok]
        for num, numZeros, numAfterZeros in parsedNum['pieces']:
            #print(numStr + durationStr, num, numZeros, numAfterZeros, durationInMilliseconds)
            millis = multiplyDecimalExact(num, numZeros, numAfterZeros, durationInMilliseconds)
            #print(millis)
            try:
                dur = str(timedelta(milliseconds=millis))
            except OverflowError:
                dur = "<overflow>"
            resultPieces.append((millis, dur, numStr + durationStr))
        results.append((totalP, resultPieces, tokens, pArr))
    return results

def parseDuration(model, topN):
    parseInfo = getParseInfo(model)
    nums = parseNumber(model, topN)
    #print("got nums", nums)
    durationTokens = parseInfo.durationTokens
    res = []
    for totalP, numStr, toks, pArr in nums:
        durationPr, durationStr, durationToks, durationPrs = parseFromTokenSet(model, 1, durationTokens)[0]
        resultP = accumulatePrs(pArr+durationPrs)
        resultStr = numStr + durationStr
        resultTokens = toks + durationToks
        resultPrs = pArr + durationPrs
        res.append(((totalP, durationPr, resultP), (numStr, durationStr), resultTokens, resultPrs))
        for t in toks:
            model.popcontext()
    return sorted(res, key=lambda x: -x[0][0])
        
        
def parseNumber(model, topN):
    parseInfo = getParseInfo(model)
    return parseFromTokenSet(model, topN, parseInfo.numberParseTokens)
        
def parseFromTokenSet(model, topN, tokens):
    toParse = SortedList()
    resultPrs = SortedList()
    
    parseInfo = getParseInfo(model)
    
    # do a greedy decode incase that's good enough
    allNumberTokens = set()
    
    backupLogits = np.array(model.logits)
    greedyTokensLists, greedyPrLists = greedyDecodeFromTokens(model, tokens, forceFirst=True, rollback=True)
    #print(greedyTokensLists, greedyPrLists)
    for greedyTokens, greedyPrs in zip(greedyTokensLists, greedyPrLists):
        pr = accumulatePrs(greedyPrs)
        #print("got greedy", pr, '"' + model.tokenizer.decode(greedyTokens) + '"', greedyTokens, greedyPrs)
        if topN == 1: # greedy is good enough
            digits = [model.detokenize([t]) for t in greedyTokens]
            resultPrs.add((-pr, "".join(digits), greedyTokens, greedyPrs))
        else:
            toParse.add((-pr, greedyTokens, greedyPrs, True))
    
    
    #print("context")
    #print(model.tokenizer.decode(model.context))
    #for t in model.context:
    #    print(t, model.idToToken(t))
    prs = parseTokenNonExclusive(model, backupLogits, tokens=tokens)
    for toks, pr in prs:
        if toks != NONTOKEN:
            #print(pr, toks, model.tokenizer.decode(toks))
            if toks[0] in [parseInfo.dashToken, parseInfo.periodToken, parseInfo.commaToken]:
                continue # the first digit should not be a special token
            toParse.add((-pr, toks, [pr], False))
    
    while len(resultPrs) < topN:
        negPr, toks, prevPrs, isDone = toParse.pop(0)
        #print(negPr, model.tokenizer.decode(toks), prevPrs, isDone)
        digits = model.detokenize(toks)
        # something like "Thousand", we need to complete it to get the prs
        didEval = False
        if len(prevPrs) < len(toks):
            model.eval(toks[:len(prevPrs)])
            unprocessedTokens = toks[len(prevPrs):]
            #print("filling in", prevPrs, len(toks))
            for t in unprocessedTokens:
                prs = npSoftmax(model.logits)
                prevPrs.append(prs[t])
                model.eval([t])
            #print("filled in", prevPrs)
            didEval = True
        if isDone:
            if didEval: # we need to add it back because it might be bad now
                toParse.add((negPr, toks, prevPrs, isDone))
            else:
                alreadyExists = False
                digitsStr = "".join(digits)
                for pr, d, t, p in resultPrs:
                    if d == digitsStr:
                        alreadyExists = True
                        break
                if not alreadyExists:
                    resultPrs.add((negPr, digitsStr, toks, prevPrs))
            continue
        #print(pr, toks, digits, prevPrs)
        if not didEval:
            model.eval(toks)
        prs = parseTokenNonExclusive(model, tokens=tokens)
        # rollback toks
        for _ in toks:
            model.popcontext()
        
        for nextTokens, nextTokensFirstPr in prs:
            allPrs = prevPrs + [nextTokensFirstPr]
            if len(nextTokens) == 1 and nextTokens[0] == parseInfo.spaceToken and toks[-1] == parseInfo.spaceToken:
                continue
                
            totalPr = accumulatePrs(allPrs)
            # pr of no more digits (i.e., we finished writing this number)
            if nextTokens == NONTOKEN:
                toParse.add((-totalPr, toks, allPrs, True)) # let it go around again, only pull off once more promising than other options
            # still going
            else:
                toParse.add((-totalPr, toks+nextTokens, allPrs, False))
    
    #print("results:\n\n")
    #for pr, digits, tokens, allPrs in resultPrs:
    #    print(digits, -pr, allPrs)
    #print("\n\n")
    return [(-pr, digits, toks, allPrs) for (pr, digits, toks, allPrs) in resultPrs]
            
    
    # todo: figure out how to get "pr of non-number" and then use that to append the pr

wordSplitterSet = set([' ', '\t', '\n', '!', ',', ';', '?', ':', '.', '¿', '¡', '…', '"', "'", '„', '“', '”', '‘', '’', '…', '(', ')', '[', ']', '{', '}', '\\', '/', '>', '<', '-', '+', '-', '*', '%', '&', '$', '@', '`', '~'])

# token contains something like ' ' at the front, or more complex things like '!', '(', etc.
def tokenStartsWithEndOfWord(model, tok):
    detokStr = model.detokenize([tok])
    firstTok = "a"
    if len(detokStr) > 0:
        firstTok = detokStr[0]
    return firstTok in wordSplitterSet

# all tokens that are like " this"
def getToksThatStartWithEndOfWord(model):
    if not hasattr(model, "toksThatStartWithEndOfWord"):
        toksThatStartWithEndOfWord = []
        for t in np.arange(model.n_vocab, dtype=np.int64):
            print(t, model.detokenize([t]))
            if tokenStartsWithEndOfWord(model=model, tok=t):
                print(t, model.detokenize([t]))
                toksThatStartWithEndOfWord.append(t)
        model.toksThatStartWithEndOfWord = np.array(toksThatStartWithEndOfWord)
    return model.toksThatStartWithEndOfWord

# This code isn't right
# At a node, I find all outgoing prs, then normalized pr divides by their total
# This is bad, consider:
# "A juniper tree"
# <Lots of other strings that don't start with A>
# What you'll get is that A has the most probability (Because it's common and simple)
#   and the remaining parts of that branch are all pr 1.0 (because it's the only option)
#   thus "A juniper" will win even if it's total pr is much less than it's competitors
# This is technically correct if we are simply constraining generation to these strings,
#   however this "early token preference" is kinda weird
# 
#  really what we want to do is compute the total pr of every string,
#    then just normalize w.r.t. that.
#  unfortunately that is too expensive
#  before I could prune via only going along paths with high enough normalized pr
#  now I need to do something slightly different
#  At node A:
#    we can get pr of each token, however each pr may be multiplied by arbitrary values
#    thus pr of each token is an upper-bound on the total pr of that branch (since all values < 1)
#    however these prs get very small, ideally we want to know total pr of all stuff so far to compare to
#    to do breadth first search, we must:
#    1. Compute pr of every option
#        For every child, compute pr of every option
#        We want to prune based on relative pr so far
#        However it could be the case the all our good candidates have very low pr end of strs
#        And thus those we pruned suddenly became good options
#        Maybe it makes sense to do a priority queue thing?
#        Keep track of current best node to expand, and expand it
#        Downside is this "hopping around" requires lots of pop and eval,
#        Thus re-doing work we already did
#    What if we just eval every string and ask for full logits? Wayy to slow   



class ParseNode(object):
    def __init__(self, parent, entireWords, tok=None):
        self.parent = parent
        self.childrenNodes = {}
        self.baseStrs = []
        self.childStrs = []
        self.entireWords = entireWords
        self.tok = tok
        
    def __lt__(self, other):
        return self.context() < other.context()
        
    def context(self):
        if not self.parent is None:
            parentContext = self.parent.context()
        else:
            parentContext = []
        if not self.tok is None:
            parentContext.append(self.tok)
        return parentContext
    
    def addToks(self, toks, baseStr):
        if len(toks) > 0:
            if not toks[0] in self.childrenNodes:
                self.childrenNodes[toks[0]] = ParseNode(parent=self, entireWords=self.entireWords, tok=toks[0])
            self.childrenNodes[toks[0]].addToks(toks=toks[1:], baseStr=baseStr)
        else:
            self.baseStrs.append(baseStr)
    
    # fetches the computed logprobs
    # you need to call compute prs first
    def getPrs(self, toks):
        if len(toks) == 0:
            if len(self.childrenNodes) == 0:
                return [], []
            else:
                # needed in case there are other children
                # this sums over prs of all tokens that aren't children
                # (or if entire words, tokens that aren't children and include some end of word thing (whitespace, punctuation) at the front)
                return [self.prOfBaseStr], [self.normalizedPrOfBaseStr]
        else:
            t = toks[0]
            prs, normalizedPrs = self.childrenNodes[t].getPrs(toks=toks[1:])
            prs = [self.childPrs[t]] + prs
            normalizedPrs = [self.normalizedChildPrs[t]] + normalizedPrs
            return prs, normalizedPrs
    
    # computes all the logprobs
    def computePrs(self, model, curNormalizedPr, curPr, minNormalizedPr=0.0, debug=False, expandChildren=True):
        if hasattr(self, "computedPrs") and self.computedPrs and not expandChildren:
            return
        else:
            self.computedPrs = True
        
        logits = np.array(model.logits)
        prs = np.array(softmax(logits))
        
        lookup = {}
        for i, t in enumerate(self.childrenNodes.keys()):
            lookup[t] = i
        
        doubleCountedChildrenInds = []
        doubleCountedChildrenToks = []
        # any leaf nodes at this position:
        if len(self.baseStrs) > 1:
            raise Exception(f"Multiple leaf nodes {self.baseStrs}, did you have a duplicate str?")
        elif len(self.baseStrs) == 1 and len(self.childrenNodes) > 0:
            if self.entireWords:
                indsOfBaseStr = getToksThatStartWithEndOfWord(model=model)
                # if entireWords flag is true we consider any token with a " spaceInFront" to add to baseStr
                # however, it's possible that one of our children also have a space in front
                # (consider "home" and "home of juniper")
                # thus, we need to note those and not double count them
                for t, i in lookup.items():
                    if tokenStartsWithEndOfWord(model=model, tok=t):
                        doubleCountedChildrenInds.append(i)
                        doubleCountedChildrenToks.append(t)            
            else:
                # all tokens that aren't child tokens
                allInds = np.zeros([model.n_vocab])
                for t in self.childrenNodes.keys():
                    allInds[t] = 1
                indsOfBaseStr = np.where(allInds==0)[0]
            logitsOfBaseStr = logits[indsOfBaseStr]
            if debug: print("argmaxes:", [(model.detokenize([t]), prs[t]) for t in np.argsort(-logits)[:10]])
            if debug: print("inds of base str", indsOfBaseStr)
            if debug: print("base pr", np.sum(prs[indsOfBaseStr]), "double counted pr", np.sum(prs[doubleCountedChildrenToks]))
            if debug: print("sum all prs", np.sum(prs))
            self.prOfBaseStr = np.sum(prs[indsOfBaseStr]) - np.sum(prs[doubleCountedChildrenToks])
            
        childLogits = np.zeros(len(lookup))
        self.childPrs = {}
        self.normalizedChildPrs = {}
        for t, child in self.childrenNodes.items():
            childLogits[lookup[t]] = logits[t]
            self.childPrs[t] = prs[t]
        del prs
        del logits
        
        if len(self.baseStrs) > 0 and len(self.childrenNodes) > 0: 
            ## manually do softmax with exp sum trick to save space
            ## maxVal = max(vals)
            maxVal = max(np.max(childLogits), np.max(logitsOfBaseStr))
            ## e_x = [math.exp(v - maxVal) for v in vals] # log exp sum trick for numerical stability
            logitsOfBaseStr -= maxVal
            childLogits -= maxVal
            np.exp(logitsOfBaseStr, out=logitsOfBaseStr) # inplace
            np.exp(childLogits, out=childLogits)
            ## total = sum(e_x)
            # subtract is to prevent double counting
            total = np.sum(logitsOfBaseStr)+np.sum(childLogits)-np.sum(childLogits[doubleCountedChildrenInds])
            ## return [(x/total) for x in e_x]
            logitsOfBaseStr /= total
            childLogits /= total
            childPrs = childLogits
            # subtract is to prevent double counting
            self.normalizedPrOfBaseStr = np.sum(logitsOfBaseStr) - np.sum(childLogits[doubleCountedChildrenInds])
                
        elif len(self.baseStrs) == 0 and len(self.childrenNodes) > 0:
            childPrs = softmax(childLogits)
        
        for t, i in lookup.items():
            self.normalizedChildPrs[t] = childPrs[i]

        if expandChildren:
            for t, child in self.childrenNodes.items():
                childNormalizedPr = curNormalizedPr*self.normalizedChildPrs[t]
                childPr = curPr*self.childPrs[t]
                if childNormalizedPr < minNormalizedPr:
                    child.fillEmptyPrs()
                else:
                    model.eval([t])
                    child.computePrs(model, curNormalizedPr=childNormalizedPr, curPr=childPr, minNormalizedPr=minNormalizedPr, debug=debug, expandChildren=expandChildren)
                    model.popcontext()
        
    def fillEmptyPrs(self):        
        if len(self.baseStrs) > 0:
            self.prOfBaseStr = 0.1
            self.normalizedPrOfBaseStr = 1.0
        
        if len(self.childrenNodes) > 0:
            self.childPrs = {}
            self.normalizedChildPrs = {}
            for t, child in self.childrenNodes.items():
                self.childPrs[t] = 0.1
                self.normalizedChildPrs[t] = 1.0
                child.fillEmptyPrs()
    

    
# do eval of context before calling this
# this will then use the logits and walk ahead a bit to get pr of each string
# todo: make eval take all tokens at once instead of one at a time, change eval logits_all=True
def prsOfStrings(model, strList, entireWords=True, minNormalizedPr=0.0, maxIters=20, debug=False):
    baseLogits = model.logits

    # remove duplicates
    strList = set(strList)
    
    strPrs = []
    # [1:] is to get rid of start of sequence token
    toksList = sorted([(tuple(model.tokenize(s)[1:]), s) for s in strList])
    
    # what do we do about blah and blahspooker?
    # we could say blah's pr is all strs that don't go to blahspooker,
    # this is valid if entireWords is False. if entireWords is True,
    # blah only counts stuff that give whitespace tokens afterwards
    print("making tree")
    # to make this cleaner we should just build a tree and then just compute prs of that
    root = ParseNode(parent=None, entireWords=entireWords)
    for toks, s in toksList:
        if debug: print("toks", toks)
        if debug: print("s", s)
        root.addToks(toks=toks, baseStr=s)
    print("done making tree")
    
    '''
    Old approach, this works but it has the problems discussed in the comments above ("A juniper" stuff)
    root.computePrs(model=model, curNormalizedPr=1.0, curPr=1.0, minNormalizedPr=minNormalizedPr)
    '''
    
    def getChildNodes(node, pr):
        node.computePrs(model, curNormalizedPr=None, curPr=pr, minNormalizedPr=None, debug=debug, expandChildren=False)
        for t, child in node.childrenNodes.items():
            yield (node.childPrs[t]*pr, child)
    
    baseContext = list(model.context)
    availableNodes = [(-1.0, root)]
    iters = 0
    while len(availableNodes) > 0 and iters < maxIters:
        pr, curNode = heapq.heappop(availableNodes)
        pr = -pr # because heapq puts smallest on top
        
        print("cur node", repr(model.detokenize(curNode.context())), "with pr", pr)
        model.eval(baseContext + curNode.context(), reset=True) # this is nice about preserving the overlapping stuff only
        
        for childPr, child in getChildNodes(node=curNode, pr=pr):
            heapq.heappush(availableNodes, (-childPr, child))   
        iters += 1
    # fill rest unfilled in stuff with low prs
    if len(availableNodes) > 0:
        for pr, curNode in availableNodes:
            curNode.fillEmptyPrs()
    # rollback to the original context
    model.eval(baseContext, reset=True)
    
    
    results = {}
    for toks, s in toksList:
        prs, normalizedPrs = root.getPrs(toks=toks)
        results[s] = {"prs": prs, "normalizedPrs": normalizedPrs, "pr": np.product(prs), "normalizedPr": np.product(normalizedPrs)}
    
    # we need to renormalize since this prunes tree paths (so values will be too large)
    if minNormalizedPr > 0.0:
        totalNormalizedPr = sum([result['normalizedPr'] for s, result in results.items()])
        totalPr = sum([result['pr'] for s, result in results.items()])
        if totalNormalizedPr == 0.0: totalNormalizedPr = 1.0
        for s, result in results.items():
            result['normalizedPr'] /= totalNormalizedPr
            result['normalizedPr2'] = result['pr']/totalPr
        
    return results


def prettyPrsOfStrings(model, context, strList, entireWords=True):
    model.reset()
    model.eval(model.tokenize(context))
    res = prsOfStrings(model=model, strList=strList, entireWords=entireWords)
    for k, results in res.items():
        print(f"{k}")
        print(f"Pr: {results['pr']}")
        print(f"NormalizedPr: {results['normalizedPr']}")
        print(" prs:", results['prs'])
        print(" normPrs:", results['normalizedPrs'])
        toks = model.tokenize(k)[1:]
        for i, t in enumerate(toks):
            print("  " + '"' + model.detokenize([t]) + '"')
            print("    pr: " + str(results['prs'][i]))
            print("   npr: " + str(results['normalizedPrs'][i]))
        # this case happens if you have overlap, like "hello" and "hello world" as two tokens
        if len(results['prs']) > len(toks):
            print("  " + "<extra>")
            print("    pr: " + str(results['prs'][len(toks)]))
            print("   npr: " + str(results['normalizedPrs'][len(toks)]))
def testPrsOfStrings(model):
    pass



def parseDigit(model):
    logits = model.logits
    parseInfo = getParseInfo(model)
    allowedTokens = parseInfo.numberTokens
    tokenLogits = [logits[x[0]] for x in allowedTokens]
    prs = softmax(tokenLogits)
    resPrs = defaultdict(lambda: 0)
    for i, (tokenI, token) in enumerate(allowedTokens):
        resPrs[token] += prs[i]
    return dict(resPrs)
    

    #= softmax(tokenLogits)

# for example numberOfInLocation("medieval fantasy world", "
def numberOfInLocation(setting, itemType, location):
    pass
    # see if there is one, or more than one ("would I expect to see more than one X?")
    
    # see if location has 


# location contents?
# todo
def makeWorldDescription(world, model, worldType, responsePrefix=""):
    return model.generateFromQ(f"Describe the lore and background of a {worldType} world in 2 sentences.", responsePrefix=responsePrefix)












def rollbackTests(model):
    model.reset()
    model.eval([1])
    model.eval([2])
    model.eval([3])
    model.eval([4])
    model.popcontext()
    model.eval([5])
    print("[1,2,3,~4,5]", hash(np.array(model.logits).data.tobytes()), np.array(model.logits)[0:4])
    
    model.reset()
    model.eval([1])
    model.eval([2])
    model.eval([3])
    model.eval([5])
    print("[1,2,3,5]", hash(np.array(model.logits).data.tobytes()), np.array(model.logits)[0:4])
    
    def randTokens(numTokens):
        return [random.randint(2, model.vocab_size-1) for _ in range(numTokens)]
    
    minSize = 4
    maxSize = 128
    for _ in range(10):
        tokens = [1] + randTokens(random.randint(minSize,maxSize))
        bonusTokens = randTokens(random.randint(minSize,maxSize))
        otherTokens = randTokens(random.randint(minSize,maxSize))
        model.reset()
        logitsAfterRollback = []
        for t in tokens + bonusTokens:
            model.eval([t])
        for _ in range(len(bonusTokens)):
            model.popcontext()
        for t in otherTokens:
            model.eval([t])
            logitsAfterRollback.append(np.array(model.logits))
        
        model.reset()
        logitsWithNoRollback = []
        for t in tokens:
            model.eval([t])
        for t in otherTokens:
            model.eval([t])
            logitsWithNoRollback.append(np.array(model.logits))
        
        for i, (rollback, noRollback) in enumerate(zip(logitsAfterRollback, logitsWithNoRollback)):
            print(tokens + otherTokens[:i+1])
            rollbackHash = hash(rollback.data.tobytes())
            nollbackHash = hash(noRollback.data.tobytes())
            print("rollback", rollbackHash, rollback[0:10])
            print("nollback", nollbackHash, noRollback[0:10])
            assert(rollbackHash == nollbackHash)
        
    for _ in range(10):
        tokens = [1] + randTokens(random.randint(minSize,maxSize))
        bonusTokens = randTokens(random.randint(minSize,maxSize))
        otherTokens = randTokens(random.randint(minSize,maxSize))
        model.reset()
        logitsAfterRollback = []
        model.eval(tokens)
        model.eval(bonusTokens)
        for _ in range(len(bonusTokens)):
            model.popcontext()
        for t in otherTokens:
            model.eval([t])
            logitsAfterRollback.append(np.array(model.logits))
        
        model.reset()
        logitsWithNoRollback = []
        model.eval(tokens)
        for t in otherTokens:
            model.eval([t])
            logitsWithNoRollback.append(np.array(model.logits))
        
        for i, (rollback, noRollback) in enumerate(zip(logitsAfterRollback, logitsWithNoRollback)):
            print(tokens + otherTokens[:i+1])
            rollbackHash = hash(rollback.data.tobytes())
            nollbackHash = hash(noRollback.data.tobytes())
            print("2rollback", rollbackHash, rollback[0:10])
            print("2nollback", nollbackHash, noRollback[0:10])
            assert(rollbackHash == nollbackHash)
            
def hashLogits(model):
    
    logitsHash = str(hash(np.array(model.logits).data.tobytes()))
    topK = [x for x in np.argsort(-np.array(model.logits))[0:5]]
    return logitsHash, topK

def smallLogitsSample(model):
    return np.array(model.logits)[0:5]


def modelTests(model):
    
    model.reset()
    model.eval([2])
    model.eval([3])
    logitsOneAtATime = np.array(model.logits)

    model.reset()
    model.eval([2,3])
    logitsAllAtOnce = np.array(model.logits)

    if not np.all(logitsAllAtOnce == logitsOneAtATime):
        print("Failure: one at a time vs all at once are not the same:")
        print("logitsOneAtATime:", logitsOneAtATime[0:5])
        print("logitsAllAtOnce:", logitsAllAtOnce[0:5])
        print("equal?", np.all(np.array(logitsOneAtATime)==np.array(logitsAllAtOnce)))
    else:
        print("Success: finished one at a time vs all at once test")

    
    # can safely backup logits
    
    model.reset()
    model.eval([2])
    logits = model.logits
    ai, bi, ci = 0, 2, 7
    a,b,c = float(logits[ai]), float(logits[bi]), float(logits[ci])
    model.eval([3])
    a2,b2,c2 = float(logits[ai]), float(logits[bi]), float(logits[ci])
    if a != a2 or b != b2 or c != c2:
        print("Failure: cannot backup logits as they seem to be a reference")
        print("please fix this as code depends on the ability to backup logits")
        print("original logit values", a, b, c)
        print("backuped logit values after doing an eval", a2, b2, c2)
    else:
        print("Success: Can safely backup logits")

    # special square is a unique token (needed to force tokenizer to not stick spaces in front of things)
    model.tokenize(specialSquare)
    allTokens = [model.detokenize([t]) for t in range(model.n_vocab)]
    tokensWithSquare = [t for t in allTokens if specialSquare in t]
    if len(tokensWithSquare) > 1:
        print("Failure: special square has more than one token:", tokensWithSquare)
        print("please use something else as special square")
    else:
        print("Success: special square has a unique token")

    # anything tokenized should have bos in front
    bos_token = model.llm.token_bos()
    out = model.tokenize("hello")
    if bos_token != out[0]:
        print(f"Failure: bos token of {bos_token} wasn't present in tokenized hello of {out}")
        print("To correct this, make sure that any string tokenized will result in a bos token at the front")
    else:
        print("Success: BOS token is at front when tokenizing")
    
     
    # "hello" should tokenize as bos_token hello_token (no space in front)
    toks = "".join([model.detokenize([t]) for t in model.tokenize("hello")[1:]])
    if " " in toks:
        print(f"Failure: hello should tokenize to 'hello' (note there is no spaces), instead we got '{toks}'")
        print("Consider a different special square")
    else:
        print("Success: space isn't added to the front when tokenizing")
    
    # " hello" should tokenize and detokenize back to " hello" (note the single space in front)
    toks = "".join([model.detokenize([t]) for t in model.tokenize(" hello")[1:]])
    if toks != ' hello':
        print(f"Failure: ' hello' should tokenize to one or two tokens with a space in the first one (note there is no spaces), instead we got '{toks}'")
    else:
        print("Success: space is added to the front in tokenization if requested")
    
    model.reset()
    toks = [2,5,10]
    numFailed = 0
    for t in toks:
        model.reset()
        model.eval([t])
        baseLogits = np.array(model.logits)
        logitsHash, logitsSample = hashLogits(model), smallLogitsSample(model)
        for i in range(10):
            model.reset()
            model.eval([t])
            logits = np.array(model.logits)
            if not np.all(baseLogits == logits):
                print("did not equal when running single token", t)
                print("first run values", logitsHash, logitsSample)
                print(f"{i}th run values", hashLogits(model), smallLogitsSample(model))
                numFailed += 1
    print("finished reproduce test with", numFailed, "failures")

    
    import itertools
    
    numFailed = 0
    
    for extra in [[], [4], [5,6], [2,6,8,22,4,1,55,43,24,22,68,33,22,33]]:
        for i,j,k in itertools.product(toks, toks, toks):
        
            extraStr = " ".join([str(x) for x in extra])
            
            # i extra k
            model.reset()
            model.eval([i])
            for l in extra:
                model.eval([l])
            model.eval([k])
            baseLogits = np.array(model.logits)
            logitsHash, logitsSample = hashLogits(model), smallLogitsSample(model)
            
            # i extra ~j k
            model.reset()
            model.eval([i])
            for l in extra:
                model.eval([l])
            model.eval([j])
            model.popcontext()
            model.eval([k])
            
            logits = np.array(model.logits)
            if not np.all(baseLogits == logits):
                print("pop context didn't work correctly for sequence", i, extraStr, j, k)
                print(f"output of     {i} {extraStr} {k}", logitsHash, logitsSample)
                print(f"output of {i} {extraStr} ~{j} {k}", hashLogits(model), smallLogitsSample(model))        
                numFailed += 1
            
            
            # i k
            model.reset()
            model.eval([i])
            model.eval([k])
            baseLogits = np.array(model.logits)
            logitsHash, logitsSample = hashLogits(model), smallLogitsSample(model)
            
            # i ~extra ~j k
            model.reset()
            model.eval([i])
            for l in extra:
                model.eval([l])
            model.eval([j])
            model.popcontext()
            for l in extra:
                model.popcontext()
            model.eval([k])
            
            extraStrInvert = " ".join(["~" + str(x) for x in extra])
            
            logits = np.array(model.logits)
            if not np.all(baseLogits == logits):
                print("pop context didn't work correctly for sequence", i, extraStr, j, k)
                print(f"output of     {i} {k}", logitsHash, logitsSample)
                print(f"output of {i} {extraStrInvert} ~{j} {k}", hashLogits(model), smallLogitsSample(model))        
                numFailed += 1            
  
            # i j k
            model.reset()
            model.eval([i])
            model.eval([j])
            model.eval([k])
            baseLogits = np.array(model.logits)
            logitsHash, logitsSample = hashLogits(model), smallLogitsSample(model)
            
            # i ~extra j ~extra k
            model.reset()
            model.eval([i])
            for l in extra:
                model.eval([l])
            for l in extra:
                model.popcontext()
            model.eval([j])
            for l in extra:
                model.eval([l])
            for l in extra:
                model.popcontext()
            model.eval([k])
            
            extraStrInvert = " ".join(["~" + str(x) for x in extra])
            
            logits = np.array(model.logits)
            if not np.all(baseLogits == logits):
                print("pop context didn't work correctly for sequence", i, j, k)
                print(f"output of     {i} {j} {k}", logitsHash, logitsSample)
                print(f"output of {i} {extraStrInvert} {j} {extraStrInvert} {k}", hashLogits(model), smallLogitsSample(model))        
                numFailed += 1            
  
  
    print("finished popcontext test with", numFailed, "failures")
    
    

def doTests(model):


    


    model.reset()
    model.eval([4])
    print('[(4)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([4])
    print('[(4)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([4])
    print('[(4)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([4])
    print('[(4)]', hashLogits(model), smallLogitsSample(model))

    model.reset()
    model.eval([4])
    print('[(4)]', hashLogits(model), smallLogitsSample(model))
    model.eval([2])
    print('[(4),(2)]', hashLogits(model), smallLogitsSample(model))
    model.popcontext()
    model.eval([2])
    print('[(4),~(2),(2)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([2])
    print('[(2)]', hashLogits(model), smallLogitsSample(model))
    model.eval([2])
    print('[(2),(2)]', hashLogits(model), smallLogitsSample(model))
    

    model.reset()
    model.eval([4])
    print('[(4)]', hashLogits(model), smallLogitsSample(model))
    model.eval([2])
    print('[(4),(2)]', hashLogits(model), smallLogitsSample(model))
    model.eval([3])
    print('[(4),(2),(3)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([4])
    print('[(4)]', hashLogits(model), smallLogitsSample(model))
    model.eval([2])
    print('[(4),(2)]', hashLogits(model), smallLogitsSample(model))
    model.eval([3])
    print('[(4),(2),(3)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([4,2,3])
    print('[(4,2,3)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([4,2,3])
    print('[(4,2,3)]', hashLogits(model), smallLogitsSample(model))
    model.eval([4])
    print('[(4,2,3),(4)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([4,2,3])
    print('[(4,2,3)]', hashLogits(model), smallLogitsSample(model))
    model.eval([4])
    print('[(4,2,3),(4)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([4,2,3,4])
    print('[(4,2,3,4)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([4,2,3,4])
    print('[(4,2,3,4)]', hashLogits(model), smallLogitsSample(model))
    
    
    model.reset()
    model.eval([1,2,3])
    print('[(1,2,3)]', hashLogits(model), smallLogitsSample(model))
    model.eval([4])
    print('[(1,2,3),(4)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([1,2,3])
    print('[(1,2,3)]', hashLogits(model), smallLogitsSample(model))
    model.eval([5])
    print('[(1,2,3),(5)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([1,2,3])
    print('[(1,2,3)]', hashLogits(model), smallLogitsSample(model))
    model.eval([4])
    print('[(1,2,3),(4)]', hashLogits(model), smallLogitsSample(model))
    model.popcontext()
    model.eval([5])
    print('[(1,2,3),~(4),(5)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([1,2,3])
    print('[(1,2,3)]', hashLogits(model), smallLogitsSample(model))
    model.eval([5])
    print('[(1,2,3),(5)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([1])
    model.eval([2])
    model.eval([3])
    print('[(1),(2),(3)]', hashLogits(model), smallLogitsSample(model))
    model.eval([4])
    print('[(1),(2),(3),(4)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([1])
    model.eval([2])
    model.eval([3])
    model.eval([4])
    print('[(1),(2),(3),(4)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([1])
    model.eval([2])
    model.eval([3])
    model.eval([5])
    print('[(1),(2),(3),(5)]', hashLogits(model), smallLogitsSample(model))
    model.popcontext()
    model.eval([4])
    print('[(1),(2),(3),~(5),(4)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([1])
    model.eval([2])
    model.eval([3])
    model.eval([4])
    model.eval([5])
    print('[(1),(2),(3),(4),(5)]', hashLogits(model), smallLogitsSample(model))
    model.eval([6])
    print('[(1),(2),(3),(4),(5),(6)]', hashLogits(model), smallLogitsSample(model))
    model.popcontext()
    model.popcontext()
    model.eval([12])
    print('[(1),(2),(3),(4),~(5),~(6),(12)]', hashLogits(model), smallLogitsSample(model))
    model.eval([13])
    print('[(1),(2),(3),(4),~(5),~(6),(12),(13)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([1])
    model.eval([2])
    model.eval([3])
    model.eval([4])
    model.eval([12])
    print('[(1),(2),(3),(4),(12)]', hashLogits(model), smallLogitsSample(model))
    model.eval([13])
    print('[(1),(2),(3),(4),(12),(13)]', hashLogits(model), smallLogitsSample(model))
    model.popcontext()
    model.popcontext()
    model.eval([5])
    print('[(1),(2),(3),(4),~(12),~(13),(5)]', hashLogits(model), smallLogitsSample(model))
    model.eval([6])
    print('[(1),(2),(3),(4),(~12),~(13),(5),(6)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([1])
    model.eval([2])
    model.eval([3])
    model.eval([4])
    print('[(1),(2),(3),(4)]', hashLogits(model), smallLogitsSample(model))
    model.eval([5])
    print('[(1),(2),(3),(4),(5)]', hashLogits(model), smallLogitsSample(model))
    model.eval([6])
    print('[(1),(2),(3),(4),(5),(6)]', hashLogits(model), smallLogitsSample(model))
    model.eval([7])
    print('[(1),(2),(3),(4),(5),(6),(7)]', hashLogits(model), smallLogitsSample(model))
    model.popcontext()
    model.eval([13])
    print('[(1),(2),(3),(4),(5),(6),~(7),(13)]', hashLogits(model), smallLogitsSample(model))
    model.eval([14])
    print('[(1),(2),(3),(4),(5),(6),~(7),(13),(14)]', hashLogits(model), smallLogitsSample(model))
    model.eval([15])
    print('[(1),(2),(3),(4),(5),(6),~(7),(13),(14),(15)]', hashLogits(model), smallLogitsSample(model))
    model.eval([16])
    print('[(1),(2),(3),(4),(5),(6),~(7),(13),(14),(15),(16)]', hashLogits(model), smallLogitsSample(model))
    model.popcontext()
    model.popcontext()
    model.eval([17])
    print('[(1),(2),(3),(4),(5),(6),~(7),(13),(14),~(15),~(16),(17)]', hashLogits(model), smallLogitsSample(model))
    model.eval([18])
    print('[(1),(2),(3),(4),(5),(6),~(7),(13),(14),~(15),~(16),(17),(18)]', hashLogits(model), smallLogitsSample(model))
    model.eval([4])
    print('[(1),(2),(3),(4),(5),(6),~(7),(13),(14),~(15),~(16),(17),(18),(4)]', hashLogits(model), smallLogitsSample(model))
    model.eval([19])
    print('[(1),(2),(3),(4),(5),(6),~(7),(13),(14),~(15),~(16),(17),(18),(4),(19)]', hashLogits(model), smallLogitsSample(model))
    for i in range(9):
        model.popcontext()
    model.eval([12])
    print('[(1),(2),(3),~(4),~(5),~(6),~(7),~(13),~(14),~(15),~(16),~(17),~(18),~(4),~(19),(12)]', hashLogits(model), smallLogitsSample(model))
    model.eval([14])
    print('[(1),(2),(3),~(4),~(5),~(6),~(7),~(13),~(14),~(15),~(16),~(17),~(18),~(4),~(19),(12),(14)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([1])
    model.eval([2])
    model.eval([3])
    model.eval([12])
    print('[(1),(2),(3),(12)]', hashLogits(model), smallLogitsSample(model))
    model.eval([14])
    print('[(1),(2),(3),(12),(14)]', hashLogits(model), smallLogitsSample(model))
    model.reset()
    model.eval([1])
    model.eval([2])
    model.eval([3])
    model.eval([4])
    model.eval([5])
    model.eval([6])
    model.eval([13])
    print('[(1),(2),(3),(4),(5),(6),(13)]', hashLogits(model), smallLogitsSample(model))
    model.eval([14])
    print('[(1),(2),(3),(4),(5),(6),(13),(14)]', hashLogits(model), smallLogitsSample(model))
    model.eval([17])
    print('[(1),(2),(3),(4),(5),(6),(13),(14),(17)]', hashLogits(model), smallLogitsSample(model))
    model.eval([18])
    print('[(1),(2),(3),(4),(5),(6),(13),(14),(17),(18)]', hashLogits(model), smallLogitsSample(model))
    model.eval([4])
    print('[(1),(2),(3),(4),(5),(6),(13),(14),(17),(18),(4)]', hashLogits(model), smallLogitsSample(model))
    model.eval([19])
    print('[(1),(2),(3),(4),(5),(6),(13),(14),(17),(18),(4),(19)]', hashLogits(model), smallLogitsSample(model))
    
    model.reset()
    model.eval([1])
    model.eval([2])
    print('[(1),(2)]', hashLogits(model), smallLogitsSample(model))
    model.eval([2])
    print('[(1),(2),(2)]', hashLogits(model), smallLogitsSample(model))
    model.eval([2])
    print('[(1),(2),(2),(2)]', hashLogits(model), smallLogitsSample(model))
    
    
    
