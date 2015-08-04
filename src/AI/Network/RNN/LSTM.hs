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
-- import Debug.Trace

import Numeric.AD

-- | The LSTM data type, with the different weights and states
data LSTMNetwork = LSTMNetwork
    { lstmSize :: !Int
    , lstmWeightsW :: !(Matrix Double)
    , lstmWeightsU :: !(Matrix Double)
    , lstmBias :: !(Vector Double)
    , lstmState :: !(Vector Double)
    , lstmOutput :: !(Vector Double)
    } deriving (Show, Read, Eq)

-- | Force evaluation instance
instance NFData LSTMNetwork where
    rnf LSTMNetwork{..} = rnf (lstmSize,lstmWeightsW,lstmWeightsU,lstmBias,lstmState,lstmOutput)

-- | Implement the network evaluation and conversions functions
instance RNNEval LSTMNetwork Int where
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
    toVector LSTMNetwork{..} = vjoin
        [ flatten lstmWeightsW, flatten lstmWeightsU, lstmBias, lstmState, lstmOutput]
    rnnsize = lstmSize

-- | Full size of a network
lstmFullSize :: FullSize Int
lstmFullSize sz = (sz * sz) * 8 + sz* 4 + sz + sz

-- | Implementation of the LSTM evaluation step without explicit matrices and vectors
-- just using lists, so we can use AD on it
lstmList :: (Num b,Floating b) => Int -> [b] -> [b] -> ([b],[b])
lstmList sz lstm is = let
    msize = sz * sz
    [mW,mU,vB,vS,vO] = takes [msize * 4,msize * 4, sz * 4,sz,sz] lstm
    z = parZipWith3 (\a b c->a+b+c) (listMProd mW is) (listMProd mU vO) vB
    [i,f,c1,o] = takes (replicate 4 sz) z
    c2 = map tanh c1
    ns = zipWith (+) (zipWith (*) c2 (map sigmoid i)) (zipWith (*) (map sigmoid f) vS)
    no = zipWith (*) (map sigmoid o) (map tanh ns)
    in (mW++mU++vB++ns++no,no)


-- | Cost calculation using list representation for AD
cost' :: (Num b,Floating b,Fractional b,Show b) => Int -> b -> [[b]] -> [[b]] -> [b] -> b
cost' sz l is os lstm = let
    (_,res) = mapAccumL (lstmList sz) lstm is
--    in - (calcMeanList $ three (last res) (last os))
--    in - (calcMeanList $ concat $ zipWith (three) res os)
    in sum $ zipWith (err l) os res
    where
      err :: (Num b,Floating b) => b -> [b] -> [b] -> b
      err le a b  = sum (zipWith (\c d -> (c- d)**2 ) a b) / le
--      one i o = zipWith (*) o (map log i)
--      two i o = zipWith (*) (map (\x->1 - x) o) (map (\x->1 - log x) i)
--      three i o= zipWith (+) (one i o) (two i o)

-- | Gradient descent learning
-- The third parameter is a call back function to monitor progress and stop the learning process if needed
learnGradientDescent :: (Monad m) => LSTMNetwork -> TrainData a Int -> (LSTMNetwork -> TrainData a Int -> Int -> m Bool) -> m LSTMNetwork
learnGradientDescent lstm td progressF = go (toList $ toVector lstm) 0
    where
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
      le = fromIntegral $ tdRecSize td
      lis = map toList (tdInputs td)
      los = map toList (tdOutputs td)
      gf = grad (cost' (tdRecSize td) (auto le) (map (map auto) lis) (map (map auto) los))

-- | RMSProp learning, as far as I can make out
-- The third parameter is a call back function to monitor progress and stop the learning process if needed
learnRMSProp :: (Monad m) => LSTMNetwork -> TrainData a Int -> (LSTMNetwork -> TrainData a Int -> Int -> m Bool) -> m LSTMNetwork
learnRMSProp lstm td progressF = go ls0 (replicate myl 0) (replicate myl 0) (replicate myl 0) 0
    where
      go ls rgs rgs2 ugs gen = do
        let rnn::LSTMNetwork = fromVector (tdRecSize td) (fromList ls)
        cont <- progressF rnn td gen
        if cont
            then do
                let
                    gs= gf ls -- gradients using AD
                    rgup = force $ zipWith (\rg g-> 0.95 * rg + 0.05 * g) rgs ls
                    rg2up = force $ zipWith (\rg2 g-> 0.95 * rg2 + 0.05 * (g ** 2)) rgs2 ls
                    ugup = force $ zipWith4 (\ud zg rg rg2 -> 0.9 * ud - 1e-4 * zg / sqrt(rg2 - rg ** 2 + 1e-4)) ugs gs rgup rg2up
                    ls2 = force $ zipWith (+) ls ugup
                go ls2 rgup rg2up ugup (gen+1)
            else return rnn
      le = fromIntegral $ tdRecSize td
      lis = map toList (tdInputs td)
      los = map toList (tdOutputs td)
      gf = grad (cost' (tdRecSize td) (auto le) (map (map auto) lis) (map (map auto) los))
      ls0 = toList $ toVector lstm
      myl= length ls0
