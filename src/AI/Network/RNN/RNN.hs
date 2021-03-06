{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards, PatternGuards, BangPatterns #-}
-----------------------------------------------------------------------------
--
-- Module      :  AI.Network.RNN.RNN
-- Copyright   :  (c) JP Moresmau
-- License     :  BSD3
--
-- Maintainer  :  JP Moresmau <jp@moresmau.fr>
-- Stability   :  experimental
-- Portability :
--
-- | Recurrent neural network
--
-----------------------------------------------------------------------------

module AI.Network.RNN.RNN  where

import Numeric.LinearAlgebra.HMatrix hiding ((|>))
import Control.DeepSeq
import Control.Monad.Random hiding (fromList)

import AI.Network.RNN.Types
import AI.Network.RNN.Util

data RNNDimensions = RNNDimensions
  { rnndInput    :: Int
  , rnndInternal :: Int
  , rnndOutput   :: Int
  , rnndWithBack :: Bool
  }  deriving (Show,Read,Eq,Ord)


instance NFData RNNDimensions where
    rnf RNNDimensions{..} = rnf (rnndInput,rnndInternal,rnndOutput,rnndWithBack)

data RNNetwork = RNNetwork
  { rnnDimensions :: RNNDimensions
  , rnnMIn        :: Matrix Double
  , rnnM          :: Matrix Double
  , rnnMOut       :: Matrix Double
  , rnnMBack      :: Maybe (Matrix Double)
  , rnnState      :: Vector Double
  , rnnOutput     :: Vector Double
  } deriving (Show,Read,Eq)

networkDimensions :: RNNetwork -> RNNDimensions
networkDimensions = rnnDimensions

instance NFData RNNetwork where
    rnf RNNetwork{..} = rnf (rnnDimensions,rnnMIn,rnnM,rnnMOut,rnnMBack,rnnState,rnnOutput)

createNetwork
    :: RNNDimensions
    -> Matrix Double
    -> Matrix Double
    -> Matrix Double
    -> Maybe (Matrix Double)
    -> Vector Double
    -> Vector Double
    -> Either [String] RNNetwork
createNetwork dim mIn m mOut mmback st out =
    case checkNetworkDimensions dim mIn m mOut mmback st of
        []  -> Right $ RNNetwork dim mIn m mOut mmback st out
        err -> Left err

createNetworkFromArray
    :: RNNDimensions
    -> [Double]
    -> Either String RNNetwork
createNetworkFromArray dim@RNNDimensions{..} v =
    let
        enough = length v >= totalDataLength dim
    in if enough
        then Right $ fromVector dim $ fromList v
        else Left $ "Not enough data in array (needs at least"++show (totalDataLength dim)++")"

networkToArray :: RNNetwork -> [Double]
networkToArray RNNetwork{..} =
       concat (toLists rnnMIn)
    ++ concat (toLists rnnM)
    ++ concat (toLists rnnMOut)
    ++ (case rnnMBack of
            Just b -> concat $ toLists b
            _      -> [])
    ++ toList rnnState
    ++ toList rnnOutput

createNetworkFromVector
    :: RNNDimensions
    -> Vector Double
    -> Either String RNNetwork
createNetworkFromVector dim@RNNDimensions{..} v =
    let
        enough = size v >= totalDataLength dim
    in if enough
        then {-# SCC "createNetworkFromVector" #-} Right $ fromVector dim v
        else Left $ "Not enough data in vector (needs at least"++ show (totalDataLength dim) ++")"


inputMatrixLength :: RNNDimensions -> Int
inputMatrixLength RNNDimensions{..} = rnndInput * rnndInternal

internalMatrixLength :: RNNDimensions -> Int
internalMatrixLength RNNDimensions{..} = rnndInternal * rnndInternal

outputMatrixLength :: RNNDimensions -> Int
outputMatrixLength RNNDimensions{..} = rnndOutput * (rnndInput + rnndInternal)

backMatrixLength :: RNNDimensions -> Int
backMatrixLength RNNDimensions{..} = if rnndWithBack then rnndInternal * rnndOutput else 0

totalDataLength :: FullSize RNNetwork
totalDataLength dim@RNNDimensions{..} =
    inputMatrixLength dim + internalMatrixLength dim + outputMatrixLength dim
    + backMatrixLength dim
    + rnndInternal + rnndOutput


randNetwork :: (Monad m,RandomGen g) =>  RNNDimensions -> RandT g m RNNetwork
randNetwork dim@RNNDimensions{..} = do
    s <- getRandom
    -- vs <- sequence (replicate (totalDataLength dim) getRandom)
    return $ fromVector dim $ randomVector s Gaussian (totalDataLength dim)

collectErrors :: [(Bool,a)] ->  [a]
collectErrors = map snd . filter fst

checkDimensions
    :: RNNDimensions
    -> [String]
checkDimensions RNNDimensions{..} = collectErrors
    [(rnndInput    < 0, "input must be >=0")
    ,(rnndInternal < 1, "internal must be >0")
    ,(rnndOutput   < 1, "output must be >0")]

checkNetworkDimensions
    :: RNNDimensions
    -> Matrix Double
    -> Matrix Double
    -> Matrix Double
    -> Maybe (Matrix Double)
    -> Vector Double
    -> [String]
checkNetworkDimensions dim@RNNDimensions{..} mIn m mOut mmback st = checkDimensions dim ++ collectErrors (
    [(cols mIn /= rnndInput, "input matrix column count doesn't match input neurons count")
    ,(rows mIn /= rnndInternal, "input matrix row count doesn't match internal neurons count")
    ,(cols m /= rnndInternal, "internal matrix column count doesn't match internal neurons count")
    ,(rows m /= rnndInternal, "internal matrix row count doesn't match internal neurons count")
    ,(cols mOut /= (rnndInput + rnndInternal), "internal matrix column count doesn't match input + internal neurons count")
    ,(rows mOut /= rnndOutput, "output matrix row count doesn't match output neurons count")
    ,(size st /= rnndInternal,"internal state length doesn't match internal neurons count")]
    ++(case mmback of
        Nothing -> []
        Just mback ->
            [(rows mback /= rnndInternal,"back matrix row count doesn't match internal neurons count")
            ,(cols m /= rnndOutput,"back matrix column count doesn't match output neurons count")
            ]))

instance RNNEval RNNetwork where
    type Size RNNetwork = RNNDimensions
    evalStep rnn@RNNetwork{..} iv =
        let
            sum1 = (rnnMIn #> iv) + (rnnM #> rnnState)
            sum2 = case rnnMBack of
                Just mback -> sum1 + (mback #> rnnOutput)
                _          -> sum1
            s2 = cmap tanh sum2
            out = cmap sigmoid (rnnMOut #> vjoin [iv,s2])
        in (rnn{rnnState=s2,rnnOutput=out},out)
    toVector RNNetwork{..} = vjoin
        [ flatten rnnMIn
        , flatten rnnM
        , flatten rnnMOut
        , maybe (fromList []) flatten rnnMBack
        , rnnState
        , rnnOutput
        ]
    fromVector dim@RNNDimensions{..} v =
        let
            lenIn   = inputMatrixLength dim
            len     = internalMatrixLength dim
            lenOut  = outputMatrixLength dim
            lenBack = backMatrixLength dim
            [v1,v2,v3,v4,v5,v6] = takesV [lenIn,len,lenOut,lenBack,rnndInternal,rnndOutput] v
            mIn     = reshape rnndInput v1
            m       = reshape rnndInternal v2
            mOut    = reshape (rnndInput + rnndInternal) v3
            mback = if rnndWithBack
                            then Just $ reshape rnndOutput v4
                            else Nothing
        in RNNetwork dim mIn m mOut mback v5 v6
    rnnsize = rnnDimensions
    fullSize = totalDataLength . rnnDimensions



