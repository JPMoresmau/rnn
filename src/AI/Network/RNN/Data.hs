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
-- |
--
-----------------------------------------------------------------------------

module AI.Network.RNN.Data  where

import qualified Data.Text as T
import qualified Data.Map as DM
import Data.Maybe
import Data.Tuple
import Data.List
import Data.Ord
import Control.Monad.Random as R
import Control.Monad
import Numeric.LinearAlgebra.HMatrix  as M
import Data.Foldable as F (toList)

import AI.Network.RNN.Types
import AI.Network.RNN.Util
import Debug.Trace


import Data.Char
import Numeric

textToTrainData :: T.Text -> TrainData (DM.Map Int Char) Int
textToTrainData t =
    let (ids,m) = T.foldl' toIDs ([],DM.empty) t
        sz = DM.size m
        is = map (toArr sz) $ reverse ids
        charMap = DM.fromList $ map swap $ DM.assocs m
    in
        TrainData ([M.fromList $ replicate sz 0] ++ init is) is (DM.size charMap) charMap
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
        toArr sz idx = M.fromList $ (replicate idx 0) ++ [1] ++ (replicate (sz-idx-1) 0)

dataToText :: DM.Map Int Char -> [Vector Double] -> T.Text
dataToText m  = T.pack . F.toList . fmap toC
    where
        toC :: Vector Double -> Char
        toC ds = fromJust $ DM.lookup (maxIndex ds) m

randDataToText :: RandomGen g => DM.Map Int Char -> [Vector Double] -> Rand g T.Text
randDataToText m  = (liftM T.pack) . mapM (toC . M.toList)
    where
        toC :: RandomGen g => [Double] -> Rand g Char
        toC ls = do
            let m1 = normalize $ map (\(ix,d)->(fromJust $ DM.lookup ix m,d)) $ zip [0..] ls
                m2 = map (\(a,b)->(a,toRational b)) m1
                s  = sum $ map (exp . snd) m1
                m3 = map (\(a,b)-> (a,exp(b)/s)) m1
            R.fromList m2
            --return $ fst $ last $ sortBy (comparing snd) m3


generate :: (RandomGen g,RNNEval a sz) => DM.Map Int Char -> Int -> Int -> a -> Rand g T.Text
generate m nb sz rnn =generate' m nb sz rnn dataToText

generateB :: (RandomGen g,RNNEval a sz) => DM.Map Int Char -> Int -> Int -> a -> Rand g T.Text
generateB m nb sz rnn =generate' m nb sz rnn dataToTextB


generate' :: (RandomGen g,RNNEval a sz) => DM.Map Int Char -> Int -> Int -> a ->
    (DM.Map Int Char -> [Vector Double] -> T.Text) -> Rand g T.Text
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

textToTrainDataB :: T.Text -> TrainData (DM.Map Int Char) Int
textToTrainDataB t =
    let (ids,m) = T.foldl' toIDs ([],DM.empty) t
        sz = binaryDigits $ DM.size m
        is = map (toArr sz) $ reverse ids
        charMap = DM.fromList $ map swap $ DM.assocs m
    in
        TrainData ([M.fromList $ replicate sz 0] ++ init is) is sz charMap
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
        toArr sz idx = let
            s = showIntAtBase 2 intToDigit idx ""
            s2 = (replicate (sz - (length s)) '0') ++ s
            in M.fromList $ map (fromIntegral . digitToInt) s2
            --M.fromList $ (replicate idx 0) ++ [1] ++ (replicate (sz-idx-1) 0)

dataToTextB :: DM.Map Int Char -> [Vector Double] -> T.Text
dataToTextB m  = T.pack . F.toList . fmap toC
    where
        toC :: Vector Double -> Char
        toC ds = let
            s = map (\d -> if d> 0.9 then '1' else '0') $ M.toList ds
            ix = fst $ head $ readInt 2 (\c->c `elem` ['0','1']) digitToInt s
            mc = DM.lookup ix m
            in case mc of
                Just c -> c
                Nothing -> fromJust $ DM.lookup 1 m
         --fromJust $ DM.lookup (maxIndex ds) m
