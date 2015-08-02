-----------------------------------------------------------------------------
--
-- Module      :  AI.Network.RNN
-- Copyright   :  (c) JP Moresmau
-- License     :  BSD3
--
-- Maintainer  :  JP Moresmau <jp@moresmau.fr>
-- Stability   :  experimental
-- Portability :
--
-- | Library API
--
-----------------------------------------------------------------------------

module AI.Network.RNN
 ( RNNEval (..)
 , RNNDimensions(..)
 , RNNetwork
 , evalSteps
 , totalDataLength
 , randNetwork
 , createNetwork
 , createNetworkFromArray
 , networkToArray
 , createNetworkFromVector
 , networkDimensions
 , cost

 , LSTMNetwork
 , lstmFullSize
 , lstmList
 , learnGradientDescent

  , textToTrainData
  , dataToText
  , generate
  , textToTrainDataS
  , dataToTextS
  , generateS

  , RNNData(..)
  , buildNetworkData
  , stopf
  , mutateNetwork

  , TrainData(..)
 ) where

import AI.Network.RNN.Data
import AI.Network.RNN.Genetic
import AI.Network.RNN.LSTM
import AI.Network.RNN.RNN
import AI.Network.RNN.Types

