import { cn } from '@/lib/utils'
import { ButtonHTMLAttributes, ReactNode } from 'react'

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  active?: boolean
  children: ReactNode
  size?: 'sm' | 'md'
}

export function IconButton({ active, children, size = 'sm', className, ...props }: Props) {
  return (
    <button
      {...props}
      className={cn(
        'inline-flex items-center justify-center rounded transition-colors',
        size === 'sm' ? 'h-7 w-7 text-sm' : 'h-8 w-8 text-base',
        active
          ? 'bg-sky-500/20 text-sky-400 ring-1 ring-sky-500/50'
          : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800',
        'disabled:opacity-40 disabled:cursor-not-allowed',
        className,
      )}
    >
      {children}
    </button>
  )
}
