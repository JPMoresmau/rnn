{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
-----------------------------------------------------------------------------
--
-- Module      :  AI.Network.RNN.Data
-- Copyright   :  (c) JP Moresmau
-- License     :  BSD3
--
-- Maintainer  :  JP Moresmau <jp@moresmau.fr>
-- Stability   :  experimental
-- Portability :
--
-- | Conversion from and to text data
--
-----------------------------------------------------------------------------

module AI.Network.RNN.Data  where

import qualified Data.Text as T
import qualified Data.Map as DM
import Data.Maybe
import Data.Tuple
import Data.List
import Control.Monad.Random as R
import Control.Monad
import Numeric.LinearAlgebra.HMatrix  as M
import Data.Foldable as F (toList)

import AI.Network.RNN.Types
import AI.Network.RNN.Util
-- import Debug.Trace


-- | Transform text into training data
-- the encoding of each Char is "one of v": each step is a vector of n values, n being the number of characters, and each character
-- is encoded as one value in the vector being 1
textToTrainData :: T.Text -> TrainData (DM.Map Int Char) Int
textToTrainData t =
    let (ids,m) = T.foldl' toIDs ([],DM.empty) t
        sz = DM.size m
        is = map (toArr sz) $ reverse ids
        charMap = DM.fromList $ map swap $ DM.assocs m
    in
        TrainData (M.fromList (replicate sz 0): init is) is (DM.size charMap) charMap
    where
        toIDs :: ([Int],DM.Map Char Int) -> Char -> ([Int],DM.Map Char Int)
        toIDs (ids,m) c =
            let mmyid = DM.lookup c m
            in case mmyid of
                Just myid -> (myid:ids,m)
                Nothing   ->
                    let myid = DM.size m
                    in (myid:ids,DM.insert c myid m)
        toArr :: Int -> Int -> Vector Double
        toArr sz idx = M.fromList $ replicate idx 0 ++ [1] ++ replicate (sz-idx-1) 0

-- | Decode the data from the one of v encoding
dataToText :: DM.Map Int Char -> [Vector Double] -> T.Text
dataToText m  = T.pack . F.toList . fmap toC
    where
        toC :: Vector Double -> Char
        toC ds = fromJust $ DM.lookup (maxIndex ds) m

-- | Generate random text with the probabilities given by the vector (not used)
randDataToText :: RandomGen g => DM.Map Int Char -> [Vector Double] -> Rand g T.Text
randDataToText m  = liftM T.pack . mapM (toC . M.toList)
    where
        toC :: RandomGen g => [Double] -> Rand g Char
        toC ls = do
            let m1 = normalize $ map (\(ix,d)->(fromJust $ DM.lookup ix m,d)) $ zip [0..] ls
                m2 = map (\(a,b)->(a,toRational b)) m1
                -- s  = sum $ map (exp . snd) m1
                -- m3 = map (\(a,b)-> (a,exp(b)/s)) m1
            R.fromList m2
            --return $ fst $ last $ sortBy (comparing snd) m3

-- | Generate text using one of v encoding and a given network
generate :: (RandomGen g,RNNEval a sz)
    => DM.Map Int Char -- ^ Character map
    -> Int  -- ^ Number of characters to generate
    -> Int -- ^ Size of the vectors for each characters
    -> a -- network
    -> Rand g T.Text
generate m nb sz rnn =generate' m nb sz rnn dataToText

-- | Generate text using sparse encoding and a given network
generateS :: (RandomGen g,RNNEval a sz)
    => DM.Map (Int,Int) Char -- ^ Character map
    -> Int  -- ^ Number of characters to generate
    -> Int -- ^ Size of the vectors for each characters
    -> a -- network
    -> Rand g T.Text
generateS m nb sz rnn =generate' m nb sz rnn dataToTextS

-- | Generate text given a decoding function
-- current implementation is not random!
generate' :: (RandomGen g,RNNEval a sz) => DM.Map b Char -> Int -> Int -> a ->
    (DM.Map b Char -> [Vector Double] -> T.Text) -> Rand g T.Text
generate' m nb sz rnn f =  do
    let (_,_,alls) = foldl' go (rnn,M.fromList $ replicate sz 0,[]) [1..nb]
    --randDataToText m alls
    return $ f m alls
    where
        go :: (RNNEval a sz) => (a,Vector Double,[Vector Double]) -> Int -> (a,Vector Double,[Vector Double])
        go (rnn1,is1,oss) _ =
            let (rnn2,os) = evalStep rnn1 is1
                os2 = norm os
            in (rnn2,os2,oss ++ [os])
        norm :: Vector Double -> Vector Double
        norm os = let
            mx = maximum $ M.toList os
            --in M.assoc (M.size os) 0 [(mx,1)]
            in M.fromList $ map (\a->if a == mx then 1 else 0) $ M.toList os

-- | Train data using sparse encoding: 2 values in the vector are set to one for each character
-- This was a test to try to reduce the size of the network but is not efficient
textToTrainDataS :: T.Text -> TrainData (DM.Map (Int,Int) Char) Int
textToTrainDataS t =
    let (ids,m) = T.foldl' toIDs ([],DM.empty) t
        sz = sparseSize $ DM.size m
        spr = DM.fromList $ zip [0..] $ sparse $ DM.size m
        is = map (toArr spr sz) $ reverse ids
        charMap = DM.fromList $ map (\(c,i)->(fromJust $ DM.lookup i spr,c)) $ DM.assocs m
    in
        TrainData (M.fromList (replicate sz 0): init is) is sz charMap
    where
        toIDs :: ([Int],DM.Map Char Int) -> Char -> ([Int],DM.Map Char Int)
        toIDs (ids,m) c =
            let mmyid = DM.lookup c m
            in case mmyid of
                Just myid -> (myid:ids,m)
                Nothing   ->
                    let myid = DM.size m
                    in (myid:ids,DM.insert c myid m)
        toArr :: DM.Map Int (Int,Int) -> Int -> Int -> Vector Double
        toArr m sz idx = let
            (i1,i2) = fromJust $ DM.lookup idx m
            in M.fromList $ replicate i1 0 ++ [1] ++ replicate (i2-i1-1) 0 ++ [1] ++ replicate (sz-i2-1) 0

-- | Generate text from sparse encoding data
dataToTextS :: DM.Map (Int,Int) Char -> [Vector Double] -> T.Text
dataToTextS m  = T.pack . F.toList . fmap toC
    where
        toC :: Vector Double -> Char
        toC ds = let
            ix1 = maxIndex ds
            ix2 = maxIndex $ accum ds const [(ix1,0)]
            tpl = if ix1 < ix2
                    then (ix1,ix2)
                    else (ix2,ix1)
            mc = DM.lookup tpl m
            in fromMaybe (fromJust $ DM.lookup (0,1) m) mc
         --fromJust $ DM.lookup (maxIndex ds) m
