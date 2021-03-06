-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Hypothesis generation script with text file input, processed line-by-line.
-- By default, this will run in interactive mode.
--
--]]

require 'fairseq'

local tnt = require 'torchnet'
local tds = require 'tds'
local argcheck = require 'argcheck'
local plstringx = require 'pl.stringx'
local data = require 'fairseq.torchnet.data'
local search = require 'fairseq.search'
local tokenizer = require 'fairseq.text.tokenizer'
local mutils = require 'fairseq.models.utils'

local cmd = torch.CmdLine()
cmd:option('-path', 'model1.th7,model2.th7', 'path to saved model(s)')
cmd:option('-beam', 1, 'search beam width')
cmd:option('-lenpen', 1,
    'length penalty: <1.0 favors shorter, >1.0 favors longer sentences')
cmd:option('-unkpen', 0,
    'unknown word penalty: <0 produces more, >0 produces less unknown words')
cmd:option('-subwordpen', 0,
    'subword penalty: <0 favors longer, >0 favors shorter words')
cmd:option('-covpen', 0,
    'coverage penalty: favor hypotheses that cover all source tokens')
cmd:option('-nbest', 1, 'number of candidate hypotheses')
cmd:option('-minlen', 1, 'minimum length of generated hypotheses')
cmd:option('-maxlen', 500, 'maximum length of generated hypotheses')
cmd:option('-input', '-', 'source language input text file')
cmd:option('-sourcedict', '', 'source language dictionary')
cmd:option('-targetdict', '', 'target language dictionary')
cmd:option('-vocab', '', 'restrict output to target vocab')
cmd:option('-visdom', '', 'visualize with visdom: (host:port)')
cmd:option('-model', '', 'model type for legacy models')
cmd:option('-aligndictpath', '', 'path to an alignment dictionary (optional)')
cmd:option('-nmostcommon', 500,
    'the number of most common words to keep when using alignment')
cmd:option('-topnalign', 100, 'the number of the most common alignments to use')
cmd:option('-freqthreshold', -1,
    'the minimum frequency for an alignment candidate in order' ..
    'to be considered (default no limit)')
cmd:option('-fconvfast', false, 'make fconv model faster')

cmd:option('-unkaligndict', '', 'path to alignment dictionary')
cmd:option('-unkmarker', '<unk>', 'unknown word marker')
cmd:option('-offset', 0, 'apply offset to attention maxima')

local config = cmd:parse(arg)

-------------------------------------------------------------------
-- Load data
-------------------------------------------------------------------
config.dict = torch.load(config.targetdict)
print(string.format('| [target] Dictionary: %d types',  config.dict:size()))
config.srcdict = torch.load(config.sourcedict)
print(string.format('| [source] Dictionary: %d types',  config.srcdict:size()))

if config.aligndictpath ~= '' then
    config.aligndict = tnt.IndexedDatasetReader{
        indexfilename = config.aligndictpath .. '.idx',
        datafilename = config.aligndictpath .. '.bin',
        mmap = true,
        mmapidx = true,
    }
    config.nmostcommon = math.max(config.nmostcommon, config.dict.nspecial)
    config.nmostcommon = math.min(config.nmostcommon, config.dict:size())
end

local unkaligndict = torch.load(config.unkaligndict)

local TextFileIterator, _ =
    torch.class('tnt.TextFileIterator', 'tnt.DatasetIterator', tnt)

local inputline

TextFileIterator.__init = argcheck{
    {name='self', type='tnt.TextFileIterator'},
    {name='path', type='string'},
    {name='transform', type='function',
        default=function(sample) return sample end},
    call = function(self, path, transform)
        function self.run()
            local fd
            if path == '-' then
                fd = io.stdin
            else
                fd = io.open(path)
            end
            return function()
                if torch.isatty(fd) then
                    io.stdout:write('> ')
                    io.stdout:flush()
                end
                local line = fd:read()
                inputline = line
                if line ~= nil then
                    return transform(line)
                elseif fd ~= io.stdin then
                    fd:close()
                end
            end
        end
    end
}

local dataset = tnt.DatasetIterator{
    iterator = tnt.TextFileIterator{
        path = config.input,
        transform = function(line)
            return {
                bin = tokenizer.tensorizeString(line, config.srcdict),
                text = line,
            }
        end
    },
    transform = function(sample)
        local source = sample.bin:view(-1, 1):int()
        local sourcePos = data.makePositions(source,
            config.srcdict:getPadIndex()):view(-1, 1)
        local sample = {
            source = source,
            sourcePos = sourcePos,
            text = sample.text,
            target = torch.IntTensor(1, 1), -- a stub
        }
        if config.aligndict then
            sample.targetVocab, sample.targetVocabMap,
                sample.targetVocabStats
                    = data.getTargetVocabFromAlignment{
                        dictsize = config.dict:size(),
                        unk = config.dict:getUnkIndex(),
                        aligndict = config.aligndict,
                        set = 'test',
                        source = sample.source,
                        target = sample.target,
                        nmostcommon = config.nmostcommon,
                        topnalign = config.topnalign,
                        freqthreshold = config.freqthreshold,
                    }
        end
        return sample
    end,
}

local model
if config.model ~= '' then
    model = mutils.loadLegacyModel(config.path, config.model)
else
    model = require(
        'fairseq.models.ensemble_model'
    ).new(config)
    if config.fconvfast then
        local nfconv = 0
        for _, fconv in ipairs(model.models) do
            if torch.typename(fconv) == 'FConvModel' then
                fconv:makeDecoderFast()
                nfconv = nfconv + 1
            end
        end
        assert(nfconv > 0, '-fconvfast requires an fconv model in the ensemble')
    end
end

local vocab = nil
if config.vocab ~= '' then
    vocab = tds.Hash()
    local fd = io.open(config.vocab)
    while true do
        local line = fd:read()
        if line == nil then
            break
        end
        -- Add word on this line together with all prefixes
        for i = 1, line:len() do
            vocab[line:sub(1, i)] = 1
        end
    end
end
local searchf = search.beam{
    ttype = model:type(),
    dict = config.dict,
    srcdict = config.srcdict,
    beam = config.beam,
    lenPenalty = config.lenpen,
    unkPenalty = config.unkpen,
    subwordPenalty = config.subwordpen,
    coveragePenalty = config.covpen,
    vocab = vocab,
}

if config.visdom ~= '' then
    local host, port = table.unpack(plstringx.split(config.visdom, ':'))
    searchf = search.visualize{
        sf = searchf,
        dict = config.dict,
        sourceDict = config.srcdict,
        host = host,
        port = tonumber(port),
    }
end

local dict, srcdict = config.dict, config.srcdict
local eos = dict:getSymbol(dict:getEosIndex())
local seos = srcdict:getSymbol(srcdict:getEosIndex())
local unk = dict:getSymbol(dict:getUnkIndex())

-- Select unknown token for reference that can't be produced by the model so
-- that the program output can be scored correctly.
local runk = unk
repeat
    runk = string.format('<%s>', runk)
until dict:getIndex(runk) == dict:getUnkIndex()

function print_r ( t )  
    local print_r_cache={}
    local function sub_print_r(t,indent)
        if (print_r_cache[tostring(t)]) then
            print(indent.."*"..tostring(t))
        else
            print_r_cache[tostring(t)]=true
            if (type(t)=="table") then
                for pos,val in pairs(t) do
                    if (type(val)=="table") then
                        print(indent.."["..pos.."] => "..tostring(t).." {")
                        sub_print_r(val,indent..string.rep(" ",string.len(pos)+8))
                        print(indent..string.rep(" ",string.len(pos)+6).."}")
                    elseif (type(val)=="string") then
                        print(indent.."["..pos..'] => "'..val..'"')
                    else
                        print(indent.."["..pos.."] => "..tostring(val))
                    end
                end
            else
                print(indent..tostring(t))
            end
        end
    end
    if (type(t)=="table") then
        print(tostring(t).." {")
        sub_print_r(t,"  ")
        print("}")
    else
        sub_print_r(t,"  ")
    end
    print()
end

for sample in dataset() do
    sample.bsz = 1
    local hypos, scores, attns = model:generate(config, sample, searchf)

    -- Print results
    local sourceString = config.srcdict:getString(sample.source:t()[1])
    sourceString = sourceString:gsub(seos .. '.*', '')
--    print('S', sourceString)
--    print('O', sample.text)
--    print_r(attns)
    for i = 1, math.min(config.nbest, config.beam) do
        local hypo = config.dict:getString(hypos[i]):gsub(eos .. '.*', '')
        
        
        local htoks = plstringx.split(hypo)
        local stoks = plstringx.split(inputline)
        
--        print('H', scores[i], hypo)
        -- NOTE: This will print #hypo + 1 attention maxima. The last one is the
        -- attention that was used to generate the <eos> symbol.
        
        local _, maxattns = torch.max(attns[i], 2)  -- attns (bsz * beam) * (targetlen * sourcelen)
        local attens = maxattns:squeeze(2):totable()
--        print('A', table.concat(attens, ' '))
        
        for j = 1, #htoks do
            if htoks[j] == config.unkmarker then
                local attn = attens[j] + config.offset
                if attn == #stoks + 1 then
                    if j == 1 then
                        htoks[j] = stoks[1]
                    else
                        htoks[j] = ''
                    end
                elseif attn < 1 or attn > #stoks + 1 then
                    io.stderr:write(string.format(
                        'Sentence %d: attention index out of bound: %d\n',
                        i, attn))
                else
                    local stok = stoks[attn]
                    if unkaligndict[stok] then
                        htoks[j] = unkaligndict[stok]
                    else
                        htoks[j] = stok
                    end
                end
            end
        end
        print('H：', plstringx.join(' ', htoks))
    end

    io.stdout:flush()
    collectgarbage()
end

