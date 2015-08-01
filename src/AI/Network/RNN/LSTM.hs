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
-- |
--
-----------------------------------------------------------------------------

module AI.Network.RNN.LSTM where

import Control.DeepSeq
import Control.Monad.Random hiding (fromList)

import Data.List

import Numeric.LinearAlgebra.HMatrix

import AI.Network.RNN.Types
import AI.Network.RNN.Util
import Debug.Trace


import Numeric.AD
import Numeric.AD.Mode.Reverse
import Numeric.AD.Newton

data LSTMNetwork = LSTMNetwork
    { lstmSize :: !Int
    , lstmWeightsW :: !(Matrix Double)
    , lstmWeightsU :: !(Matrix Double)
    , lstmBias :: !(Vector Double)
    , lstmState :: !(Vector Double)
    , lstmOutput :: !(Vector Double)
    } deriving (Show, Read, Eq)

instance NFData LSTMNetwork where
    rnf LSTMNetwork{..} = rnf (lstmSize,lstmWeightsW,lstmWeightsU,lstmBias,lstmState,lstmOutput)

instance RNNEval LSTMNetwork Int where
    evalStep n@LSTMNetwork{..} is =
        let z = (lstmWeightsW #> is) + (lstmWeightsU #> lstmOutput) + lstmBias
            [i,f,c1,o] = takesV (replicate 4 lstmSize) z
            c2 = cmap tanh c1
            ns = (c2 * (cmap sigmoid i)) + ((cmap sigmoid f) * lstmState)
            no = (cmap sigmoid o) * (cmap tanh ns)
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

lstmFullSize :: FullSize Int
lstmFullSize sz = (sz * sz) * 8 + sz* 4 + sz + sz

-- randomLSTM :: (Monad m,RandomGen g) => Int -> RandT g m LSTMNetwork
-- randomLSTM sz = createRandomNetwork sz (fullSize sz)

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

cost' :: (Num b,Floating b,Fractional b,Show b) => Int -> b -> [[b]] -> [[b]] -> [b] -> b
cost' sz l is os lstm = let
    (_,res) = mapAccumL (lstmList sz) lstm is
--    in - (calcMeanList $ three (last res) (last os))
--    in - (calcMeanList $ concat $ zipWith (three) res os)
    in ((sum $ zipWith err os res))
    where
      err :: (Num b,Floating b) => [b] -> [b] -> b
      err a b    = (sum $ zipWith (\c d -> (c- d)**2 ) a b) / fromIntegral (length a)
      one i o = zipWith (*) o (map log i)
      two i o = zipWith (*) (map (\x->1 - x) o) (map (\x->1 - log x) i)
      three i o= zipWith (+) (one i o) (two i o)


learnGradientDescent :: (Monad m) => LSTMNetwork -> TrainData a Int -> (LSTMNetwork -> TrainData a Int -> Int -> m Bool) -> m LSTMNetwork
learnGradientDescent lstm td progressF = go (toList $ toVector lstm) 0
    where
      go ls gen = do
        let rnn::LSTMNetwork = fromVector (tdRecSize td) (fromList ls)
        cont <- progressF rnn td gen
        if cont
            then do
                let
                    gs= gf ls
                    ls2 = zipWith (\o g->o-g*0.1) ls gs
                go ls2 (gen+1)
            else return rnn
      le = fromIntegral $ length (tdInputs td)
      -- ale :: forall s. Reverse s Double
      -- ale = auto le
      lis = map toList (tdInputs td)
      --alis = map (map auto) lis
      los = map toList (tdOutputs td)
      -- alos = map (map auto) los
      gf = grad (cost' (tdRecSize td) (auto le) (map (map auto) lis) (map (map auto) los))

--learnGradientDescent :: (Monad m) => LSTMNetwork -> TrainData a -> (LSTMNetwork -> TrainData a -> Int -> m Bool) -> m LSTMNetwork
--learnGradientDescent lstm td progressF = do
--    let ls=toList $ toVector lstm
--        res = gradientDescent (cost' (tdRecSize td) (auto le) (map (map auto) lis) (map (map auto) los)) ls
--    go res 0
--    where
--      go (ls:ls2) gen = do
--         let rnn::LSTMNetwork = fromVector (tdRecSize td) (fromList ls)
--         cont <- progressF rnn td gen
--         if cont
--            then go ls2 (gen+1)
--            else return rnn
--      le = fromIntegral $ length (tdInputs td)
--      lis = map toList (tdInputs td)
--      los = map toList (tdOutputs td)
