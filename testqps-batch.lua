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
require 'socket'
local tnt = require 'torchnet'
local tds = require 'tds'
local argcheck = require 'argcheck'
local plstringx = require 'pl.stringx'
local data = require 'fairseq.torchnet.data'
local search = require 'fairseq.search'
local tokenizer = require 'fairseq.text.tokenizer'
local mutils = require 'fairseq.models.utils'
local lmc = require 'fairseq.text.lm_corpus'

local cmd = torch.CmdLine()

cmd:option('-path', 'model1.th7,model2.th7', 'path to saved model(s)')
cmd:option('-beam', 10, 'search beam width')
cmd:option('-lenpen', 1,
    'length penalty: <1.0 favors shorter, >1.0 favors longer sentences')
cmd:option('-unkpen', 0,
    'unknown word penalty: <0 produces more, >0 produces less unknown words')
cmd:option('-subwordpen', 0,
    'subword penalty: <0 favors longer, >0 favors shorter words')
cmd:option('-covpen', 0,
    'coverage penalty: favor hypotheses that cover all source tokens')
cmd:option('-nbest', 5, 'number of candidate hypotheses')
cmd:option('-minlen', 1, 'minimum length of generated hypotheses')
cmd:option('-maxlen', 500, 'maximum length of generated hypotheses')
cmd:option('-maxsourcelen', 50, 'maximum length of source sentences')
cmd:option('-batchsize', 5, 'batchsize')
cmd:option('-input', '-', 'source language input text file')
cmd:option('-sourcedict', 'dict.x.th7', 'source language dictionary')
cmd:option('-targetdict', 'dict.y.th7', 'target language dictionary')
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
cmd:option('-datadir', 'data-bin')

cmd:option('-unkaligndict', '', 'path to alignment dictionary')
cmd:option('-unkmarker', '<unk>', 'unknown word marker')
cmd:option('-offset', 0, 'apply offset to attention maxima')
cmd:option('-sourcepath', '', 'source file path.')

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

--function transform1(line)
--    return {
--        bin = tokenizer.tensorizeString(line, config.srcdict),
--        text = line,
--    }
--end

function transform1(line)
    return tokenizer.tensorizeString(line, config.srcdict)
end

function transform2(sample)
    local source = sample:int()
    local sourcePos = makePositions(source,
        config.srcdict:getPadIndex())
    local sample = {
        source = source,
        sourcePos = sourcePos,
        text = sample.text,
        target = torch.IntTensor(1, 1), -- a stub
    }
    return sample
end

function makePositions(source, pad)
   sourcePos = torch.Tensor(source:size()[1], source:size()[2]):fill(config.srcdict:getPadIndex())
   for i=1, source:size()[2] do
       for j=1, source:size()[1] do
           if source[j][i] ~= config.srcdict:getPadIndex() then
               sourcePos[j][i] = pad + j
           end
       end
   end
   return sourcePos
end


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

function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    local pfile = popen('ls -a "'..directory..'"')
    for filename in pfile:lines() do
        i = i + 1
        t[i] = filename
    end
    pfile:close()
    return t
end

function getinput_data(data_dir, suffix)
    file_list = scandir(data_dir)
    data_path = ""
    for i, fn in ipairs(file_list) do
        if fn and fn:sub(-string.len(suffix)) == suffix then
           data_path = data_dir .. '/' .. fn
           return data_path
        end
    end
    return data_path
end

local outputpath = config.sourcepath .. '.output'
local fileoutput = io.open(outputpath, 'w')

function fix_sent(sample, lines)
    sample.bsz = config.batchsize
    local hypos, scores, attns = model:generate(config, sample, searchf)
    print('hypos:')
    print_r(hypos)
    print("attns")
    print_r(attns)
    -- Print results
    for k = 1, sample.bsz do
        local base = (k - 1) * config.batchsize
        print("stoks")
        print_r(lines[k])
        for i = 1, math.min(config.nbest, config.beam) do
            local hypo = config.dict:getString(hypos[base + i]):gsub(eos .. '.*', '')

            local htoks = plstringx.split(hypo)
            local stoks = plstringx.split(lines[k])
            
            -- NOTE: This will print #hypo + 1 attention maxima. The last one is the
            -- attention that was used to generate the <eos> symbol.
            
            local _, maxattns = torch.max(attns[base + i], 2)  -- attns (bsz * beam) * (targetlen * sourcelen)
            local attens = maxattns:squeeze(2):totable()
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
            local res = {}
            table.insert(res, plstringx.join(' ', htoks))
            scores_cut = string.format("%.6f", scores[base + i])
            fileoutput:write(plstringx.join(' ', htoks) .. '\t' .. scores_cut .. '\n')
        end
    end
end

function generate(lines, maxlen, starttime)
    config.batchsize = #lines
    local queries = torch.Tensor(config.batchsize, maxlen + 1):fill(config.srcdict:getPadIndex())
    for k, v in pairs(lines) do
        local query = transform1(v)
        for i=1, query:size()[1] do
            queries[k][i] = query[i]
        end
    end
    print('****queries****')
    print_r(queries)
    queries = queries:transpose(1, 2)
    queries = transform2(queries)
    print('****transqueries****')
    print_r(queries)
    local time1 = socket.gettime()
    print(string.format("Spentting time %s seconds!", (time1 - starttime)))
    fix_sent(queries, lines)
    local time2 = socket.gettime()
    print(string.format("Spentting time %s seconds!", (time2 - starttime)))
    queries = {}
end

print(string.format("[%s] Starting!", os.date()))
local starttime = socket.gettime()
local cnt = 0
local lines = {}
local maxlen = 0
for line in io.lines(config.sourcepath) do
    _,n = line:gsub("%S+","")
    if n > config.maxsourcelen then
        goto continue
    end
    lines[#lines + 1] = line.strip()
    if maxlen < n then
        maxlen = n
    end
    cnt = cnt + 1
    if cnt % config.batchsize == 0 then
        generate(lines, maxlen, starttime)
        lines = {}
        maxlen = 0
    end
    ::continue::
end

if #lines ~= 0 then
    generate(lines, maxlen, starttime)
    lines = {}
    maxlen = 0
end

local endtime = socket.gettime()
print(string.format("[%s] Ending!", os.date()))
print(string.format("Spentting time %s seconds!", (endtime - starttime)))
print(string.format("QPS: %s", cnt / (endtime - starttime)))
fileoutput:close()
