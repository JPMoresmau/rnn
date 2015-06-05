{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
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
import Control.Monad.Random

import Data.List

import Numeric.LinearAlgebra.HMatrix

import AI.Network.RNN.Types
import AI.Network.RNN.Util

data LSTMNetwork = LSTMNetwork
    { lstmSize :: Int
    , lstmWeightsW :: Matrix Double
    , lstmWeightsU :: Matrix Double
    , lstmBias :: Vector Double
    , lstmState :: Vector Double
    , lstmOutput :: Vector Double
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
        in (n{lstmState=ns,lstmOutput=no},no)
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
