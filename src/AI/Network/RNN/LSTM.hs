{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE RankNTypes #-}
-----------------------------------------------------------------------------
--
-- Module      :  AI.Network.RMM.LSTM
-- Copyright   :  (c) JP Moresmau
-- License     :  BSD3
--
-- Maintainer  :  JP Moresmau <jp@moresmau.fr>
-- Stability   :  experimental
-- Portability :
--
-- | Long Short Term Memory networks
--
-----------------------------------------------------------------------------

module AI.Network.RNN.LSTM where

import Control.DeepSeq


import Data.List

import Numeric.LinearAlgebra.HMatrix

import AI.Network.RNN.Types
import AI.Network.RNN.Util
import AI.Network.RNN.Expr
import Debug.Trace
import qualified Data.IntMap as I

import Numeric.AD

-- | The LSTM data type, with the different weights and states
data LSTMNetwork = LSTMNetwork
    { lstmSize :: !Int
    , lstmWeightsW :: !(Matrix Double)
    , lstmWeightsU :: !(Matrix Double)
    , lstmBias :: !(Vector Double)
    , lstmState :: !(Vector Double)
    , lstmOutput :: !(Vector Double)
    } deriving (Show,Read,Eq)

--instance Show LSTMNetwork where
--     show s = let
--        a= show $ (rnnsize s,toVector s)
--        in trace (show $ length a) a
--
--instance Read LSTMNetwork where
--    readsPrec d s =
--        let [(ok,r)] = readsPrec d s
--            dat = uncurry fromVector ok
--        in [(dat,r)]

-- | Force evaluation instance
instance NFData LSTMNetwork where
    rnf LSTMNetwork{..} = rnf (lstmSize,lstmWeightsW,lstmWeightsU,lstmBias,lstmState,lstmOutput)

-- | Implement the network evaluation and conversions functions
instance RNNEval LSTMNetwork where
    type Size LSTMNetwork = Int
    evalStep n@LSTMNetwork{..} is =
        let z = (lstmWeightsW #> is) + (lstmWeightsU #> lstmOutput) + lstmBias
            [i,f,c1,o] = takesV (replicate 4 lstmSize) z
            c2 = cmap tanh c1
            ns = (c2 * cmap sigmoid i) + (cmap sigmoid f * lstmState)
            no = cmap sigmoid o * cmap tanh ns
        in force (n{lstmState=ns,lstmOutput=no},no)
    fromVector sz vs =
        let
            msize = sz * sz
            [v1,v2,v3,v4,v5] = takesV [msize * 4,msize * 4, sz * 4,sz,sz] vs
            m1 = reshape sz v1
            m2 = reshape sz v2
        in LSTMNetwork sz m1 m2 v3 v4 v5
    toVector LSTMNetwork{..} = force $ vjoin
        [ flatten lstmWeightsW, flatten lstmWeightsU, lstmBias, lstmState, lstmOutput]
    rnnsize = lstmSize
    fullSize = lstmFullSize .lstmSize

-- | Full size of a network
lstmFullSize :: FullSize LSTMNetwork
lstmFullSize sz = (sz * sz) * 8 + sz* 4 + sz + sz

data LSTMIO = LSTMIO
    { lioInput :: Matrix Double
    , lioLstms :: [LSTMNetwork]
    , lioOutput :: Matrix Double
    , lioSer   :: Maybe (Vector Double)
    } deriving (Eq)

instance NFData LSTMIO where
  rnf LSTMIO{..} = rnf (lioInput,lioLstms,lioOutput)

instance RNNEval LSTMIO  where
    type Size LSTMIO = (Int,Int,Int,Int)
    evalStep io@LSTMIO{..} is =
        let inps = cmap sigmoid $ lioInput #> (vjoin [is,scalar 1])
            (lstm2,outs) = evalStep lioLstms inps
        in (io{lioLstms=lstm2,lioSer=Nothing},lioOutput #> outs)
    fromVector (is,lnnSize,lnnNumber,os) vs =
        let [v1,v2,v3] = takesV [(is+1)* lnnSize ,lnnNumber * (lstmFullSize lnnSize),os * lnnSize ] vs
            m1 = reshape (is+1) v1
            m2 = reshape lnnSize v3
        in LSTMIO m1 (fromVector (replicate lnnNumber lnnSize) v2) m2 $ Just vs
    toVector LSTMIO{..} = case lioSer of
        Just vs -> vs
        Nothing -> vjoin [flatten lioInput,toVector lioLstms,flatten lioOutput]
    rnnsize LSTMIO{..} = (cols lioInput - 1, rnnsize $ head lioLstms, length lioLstms,rows lioOutput)
    fullSize =lstmioFullSize . rnnsize
--        let (r1,c1)=size lioInput
--            (r2,c2)=size lioOutput
--        in (r1*c1) + (fullSize lioLstms) + (r2*c2)

instance Show LSTMIO where
    show l = show (rnnsize l,toVector l)

instance Read LSTMIO where
    readsPrec d s =
        let [(ok,r)] = readsPrec d s
            dat = uncurry fromVector ok
        in [(dat,r)]

lstmioFullSize :: FullSize LSTMIO
lstmioFullSize (is,lnnSize,lnnNumber,os) = (is+1) * lnnSize + lnnNumber * (lstmFullSize lnnSize) + os * lnnSize

data LSTMList = LSTMList
    { llSize :: (Int,Int,Int,Int)
    , llData :: [Double]
    } deriving (Show, Read, Eq)

instance NFData LSTMList where
  rnf LSTMList{..} = rnf (llSize,llData)

instance RNNEval LSTMList where
    type Size LSTMList = (Int,Int,Int,Int)
    rnnsize = llSize
    fullSize =lstmioFullSize . rnnsize
    toDList = llData
    fromDList sz = LSTMList sz
    toVector = fromList . toDList
    fromVector sz = fromDList sz . toList
    evalStep l@LSTMList{..} is =
        let lstmio :: LSTMIO = fromDList llSize llData
            (io2,out) = evalStep lstmio is
        in (l{llData=toDList io2},out)


-- | Implementation of the LSTM evaluation step without explicit matrices and vectors
-- just using lists, so we can use AD on it
lstmList :: (Num b,Floating b) => Int -> [b] -> [b] -> ([b],[b])
lstmList sz lstm is = let
    msize = sz * sz
    [mW,mU,vB,vS,vO] = takes [msize * 4,msize * 4, sz * 4,sz,sz] lstm
    z = zipWith3 (\a b c->a+b+c) (listMProd mW is) (listMProd mU vO) vB
    [i,f,c1,o] = takes (replicate 4 sz) z
    c2 = map tanh c1
    ns = zipWith (+) (zipWith (*) c2 (map sigmoid i)) (zipWith (*) (map sigmoid f) vS)
    no = zipWith (*) (map sigmoid o) (map tanh ns)
    in (mW++mU++vB++ns++no,no)

-- | Cost calculation using list representation for AD
cost' :: (Num b,Floating b,Fractional b) => Int -> [[b]] -> [[b]] -> [b] -> b
cost' sz is os lstm = let
    (_,res) = mapAccumL (lstmList sz) lstm is
--    in - (calcMeanList $ three (last res) (last os))
--    in - (calcMeanList $ concat $ zipWith (three) res os)
    in sum $ zipWith err os res
    where
      err :: (Num b,Floating b) => [b] -> [b] -> b
      err a b  = sum (zipWith (\c d -> (c- d)**2 ) a b)
--      one i o = zipWith (*) o (map log i)
--      two i o = zipWith (*) (map (\x->1 - x) o) (map (\x->1 - log x) i)
--      three i o= zipWith (+) (one i o) (two i o)

-- | Gradient descent learning
-- The third parameter is a call back function to monitor progress and stop the learning process if needed
learnGradientDescent :: (Monad m) => LSTMNetwork -> TrainData a -> (LSTMNetwork -> TrainData a -> Int -> m Bool) -> m LSTMNetwork
learnGradientDescent lstm td progressF =  go (toList $ toVector lstm) 0
    -- go2 gds 0
    where
      go2 (ls:lss) gen =  do
        let rnn::LSTMNetwork = fromVector (tdRecSize td) (fromList ls)
        cont <- progressF rnn td gen
        if cont
            then go2 lss (gen+1)
            else return rnn
      go ls gen = do
        let rnn::LSTMNetwork = fromVector (tdRecSize td) (fromList ls)
        cont <- progressF rnn td gen
        if cont
            then do
                let
                    gs= gf ls -- gradients using AD
                    ls2 = zipWith (\o g->o-g*0.1) ls gs
                go ls2 (gen+1)
            else return rnn
      lis = map toList (tdInputs td)
      los = map toList (tdOutputs td)
      gf = grad (cost' (tdRecSize td) (map (map auto) lis) (map (map auto) los))
      gds = gradientDescent (cost' (tdRecSize td) (map (map auto) lis) (map (map auto) los)) (toList $ toVector lstm)

-- | Gradient descent learning using symbolic differentiation
--   The goal was to calculate the derivative once and then just close and eval the result using the current data
--   However it is much slower than normal automatic differentiation
learnGradientDescentSym :: (Monad m) => LSTMNetwork -> TrainData a -> (LSTMNetwork -> TrainData a -> Int -> m Bool) -> m LSTMNetwork
learnGradientDescentSym lstm td progressF =  go (toList $ toVector lstm) 0
    where
      go ls gen = do
        let rnn::LSTMNetwork = fromVector (tdRecSize td) (fromList ls)
        cont <- progressF rnn td gen
        if cont
            then do
                let
                    i = I.fromList $ zip [0..] ls
                    cexpr = map (\g-> close g i) gf
                    gs = map eval cexpr
                    ls2 = zipWith (\o g->o-g*0.1) ls gs
                go ls2 (gen+1)
            else return rnn
      lis = map toList (tdInputs td)
      los = map toList (tdOutputs td)
      ls0 = toList $ toVector lstm
      gf = map fullSimplify $ grad (cost'
            (tdRecSize td)
            (map (map (\x -> autoEval (Lit $ Lit x) I.empty)) lis)
            (map (map (\x -> autoEval (Lit $ Lit x) I.empty)) los))
            (zipWith (\_ i->Var i) ls0 [0..])

-- | RMSProp learning, as far as I can make out
-- The third parameter is a call back function to monitor progress and stop the learning process if needed
learnRMSProp :: (Monad m) => LSTMNetwork -> TrainData a -> (LSTMNetwork -> TrainData a -> Int -> m Bool) -> m LSTMNetwork
learnRMSProp lstm td progressF = go ls0 (replicate myl 0) (replicate myl 0) (replicate myl 0) 0
    where
      go ls rgs rgs2 ugs gen = do
        let rnn::LSTMNetwork = fromVector (tdRecSize td) (fromList ls)
        cont <- progressF rnn td gen
        if cont
            then do
                let
                    gs= gf ls -- gradients using AD
                    rgup = zipWith (\rg g-> 0.95 * rg + 0.05 * g) rgs ls
                    rg2up = zipWith (\rg2 g-> 0.95 * rg2 + 0.05 * (g ** 2)) rgs2 ls
                    ugup = zipWith4 (\ud zg rg rg2 -> 0.9 * ud - 1e-4 * zg / sqrt(rg2 - rg ** 2 + 1e-4)) ugs gs rgup rg2up
                    ls2 = zipWith (+) ls ugup
                go (force ls2) (force rgup) (force rg2up) (force ugup) (gen+1)
            else return rnn
      lis = map toList (tdInputs td)
      los = map toList (tdOutputs td)
      gf = grad (cost' (tdRecSize td) (map (map auto) lis) (map (map auto) los))
      ls0 = toList $ toVector lstm
      myl= length ls0
